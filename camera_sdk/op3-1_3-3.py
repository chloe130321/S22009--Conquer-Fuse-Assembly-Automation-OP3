# OP2 cameras (op2_3, op2_6) using HuarayCameraController (IMV SDK) + PLC + ConvNeXt
import os, json, cv2, torch, threading, time, traceback
import numpy as np
from datetime import datetime, timedelta  # ⭐️ MODIFIED
import shutil  # ⭐️ ADDED
from PIL import Image  # ⭐️ ADDED
from typing import Tuple

from plc_socket import plc_socket
from logger import loginfo
from torchvision import transforms
from model_setup import get_model
# from inference import predict_image  # ⭐️ REMOVED (we predict in-memory)
from camera_controller import HuarayCameraController
import torch.nn.functional as F #用於計算softmax

# =========================
# PLC socket
# =========================
# ⭐️ REMOVED: 全域的 socket_op3 已被移除
# socket_op3 = plc_socket("192.168.162.160", 8501)

# =========================
# Cameras — IMV SDK indexes (from your check.py on THIS PC)
# Adjust index if camera order changes.
# =========================
CONFIG_PATH = "config.json"
DEFAULT_CONF_THRESHOLD = 0.80

cameras = {
    "op3_1": {
        "index": 1,
        "serial": "DA26269AAK00007",
        "base_dir": r"C:\3-1_3-3\OP3-1",
        "plc_trigger": "DM4020",   # ✅ matches op3_app.py
        "plc_result": "DM4030",
        "plc_ip": "192.168.162.160", # ⭐️ ADDED
        "plc_port": 8501, # ⭐️ ADDED
        # "socket": socket_op3, # ⭐️ REMOVED
    },
    "op3_3": {
        "index": 2,
        "serial": "DA26269AAK00010",
        "base_dir": r"C:\3-1_3-3\OP3-3",
        "plc_trigger": "DM4000",   # ✅ matches op3_app.py
        "plc_result": "DM4010",
        "plc_ip": "192.168.162.160", # ⭐️ ADDED
        "plc_port": 8501, # ⭐️ ADDED
        # "socket": socket_op3, # ⭐️ REMOVED
    },
}

def load_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"⚠️ Config file {CONFIG_PATH} not found! Using defaults.")
        return {"confidence_threshold": DEFAULT_CONF_THRESHOLD, "models": {}}
    
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            print(f"✅ Config loaded: Threshold={cfg.get('confidence_threshold')}")
            return cfg
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return {"confidence_threshold": DEFAULT_CONF_THRESHOLD, "models": {}}

# 程式啟動時先讀取一次配置
APP_CONFIG = load_config()
CONF_THRESHOLD = APP_CONFIG.get("confidence_threshold", DEFAULT_CONF_THRESHOLD)
MODEL_BASE_DIR = APP_CONFIG.get("model_base_dir", r"C:\3-1_3-3\model")

# 從 config 取得檔名，若無則使用預設值或報錯
model_name_op3_1 = APP_CONFIG.get("models", {}).get("op3_1", "default_model_1.pth")
model_name_op3_3 = APP_CONFIG.get("models", {}).get("op3_3", "default_model_2.pth")

CLASSIFY_CFG = {
    "op3_1": {
        "type": "double",
        "model_path": os.path.join(MODEL_BASE_DIR, model_name_op3_1),
        "class_names": ["ok", "ng"],
        # "crop_ratio": 0.4,  <-- 刪除這行，不需要了
        "crops": [
            # 第一個裁切框 (例如：左邊的物件)
            {"x": 250, "y": 580, "w": 180, "h": 130}, 
            # 第二個裁切框 (例如：右邊的物件)
            {"x": 680, "y": 410, "w": 160, "h": 130},
        ],
    },
    "op3_3": {
        "type": "double",
        "model_path": os.path.join(MODEL_BASE_DIR, model_name_op3_3),
        "class_names": ["ok", "ng"],
        # "crop_ratio": 0.5, <-- 刪除這行
        "crops": [
            # 第一個裁切框
            {"x": 340, "y": 580, "w": 170, "h": 130},
            # 第二個裁切框
            {"x": 790, "y": 700, "w": 190, "h": 140},
        ],
    },
}




VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# =========================
# Load models once per unique path
# =========================
_MODEL_CACHE = {}
for cam, cfg in CLASSIFY_CFG.items():
    path = cfg["model_path"]
    names = cfg["class_names"]
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 1. 先讀取權重檔 (State Dict)
        state = torch.load(path, map_location=device)
        
        # 2. 建立基礎 ConvNeXt 模型
        model = models.convnext_tiny(weights=None)
        num_ftrs = model.classifier[2].in_features
        
        # 3. 智慧判斷結構：檢查權重檔是否包含 Dropout 層的特徵 ('classifier.2.1')
        has_dropout_layer = any("classifier.2.1" in k for k in state.keys())
        
        if has_dropout_layer:
            loginfo("ConvNeXtInit", f"[{cam}] 偵測到新版模型結構 (含 Dropout)")
            model.classifier[2] = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, len(names))
            )
        else:
            loginfo("ConvNeXtInit", f"[{cam}] 偵測到舊版模型結構 (不含 Dropout)")
            model.classifier[2] = nn.Linear(num_ftrs, len(names))
            
        # 4. 載入權重並設為評估模式
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        
        _MODEL_CACHE[path] = model
        loginfo("ConvNeXtInit", f"[{cam}] Loaded model: {path}")
    except Exception as e:
        loginfo("ConvNeXtInit", f"[{cam}] FAILED to load {path}: {e}")
# =========================
# Helpers
# =========================

def cleanup_old_folders(base_dir, days_to_keep):
    """⭐️ ADDED: 刪除超過指定天數的舊資料夾 (格式 YYYY-MM-DD)"""
    if not os.path.exists(base_dir):
        return  # 根目錄不存在，無需清理
    try:
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for folder_name in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder_name)
            
            if os.path.isdir(folder_path):
                try:
                    folder_date = datetime.strptime(folder_name, '%Y-%m-%d')
                    if folder_date.date() < cutoff_date.date():
                        shutil.rmtree(folder_path)
                        loginfo("Cleanup", f"[{os.path.basename(base_dir)}] 已刪除舊資料夾: {folder_path}")
                except ValueError:
                    pass
                except Exception as e:
                    loginfo("Cleanup", f"刪除資料夾 {folder_path} 時出錯: {e}")
    except Exception as e:
        loginfo("Cleanup", f"清理程序 {base_dir} 出錯: {e}")


def ensure_dirs(base_dir: str) -> Tuple[str, str]:
    date_dir = datetime.now().strftime("%Y-%m-%d")
    full_dir  = os.path.join(base_dir, date_dir)
    os.makedirs(full_dir, exist_ok=True)
    return date_dir, full_dir

def _imwrite_png(path: str, img: np.ndarray) -> bool:
    if img is None or img.size == 0:
        return False
    img = np.ascontiguousarray(img)
    ok = cv2.imwrite(path, img)
    if not ok:
        print(f"❌ imwrite failed → {path} (shape={None if img is None else img.shape}, dtype={None if img is None else img.dtype})")
    return ok

def save_full_frame(image_bgr: np.ndarray, base_dir: str, date_dir: str, base_name: str) -> Tuple[bool, str]:
    path = os.path.join(base_dir, date_dir, f"{base_name}.png")
    ok = _imwrite_png(path, image_bgr)
    return ok, path

def _center_shift_crop(img_bgr: np.ndarray, crop_ratio: float, dx: int, dy: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    cw, ch = max(1, int(w * crop_ratio)), max(1, int(h * crop_ratio))
    cx, cy = w // 2 + int(dx), h // 2 + int(dy)
    x1 = max(0, cx - cw // 2)
    y1 = max(0, cy - ch // 2)
    x2 = min(w, x1 + cw)
    y2 = min(h, y1 + ch)
    x1 = max(0, x2 - cw)
    y1 = max(0, y2 - ch)
    return img_bgr[y1:y2, x1:x2]

def _predict_in_memory(cv2_img_bgr: np.ndarray, model, class_names, transform, threshold: float) -> str:
    """⭐️ MODIFIED: 新增 threshold 參數"""
    try:
        img_rgb = cv2.cvtColor(cv2_img_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        input_tensor = transform(pil_image)
        input_batch = input_tensor.unsqueeze(0)
        device = next(model.parameters()).device
        input_batch = input_batch.to(device)
        
        with torch.no_grad():
            output = model(input_batch)
            
            # ⭐️ MODIFIED: 轉換為機率 (0~1)
            probs = F.softmax(output, dim=1) 
            
            # 取得最大機率與對應的類別索引
            top_p, top_class_idx = probs.topk(1, dim=1)
            
            conf_score = top_p.item()
            pred_index = top_class_idx.item()
            pred_class = class_names[pred_index]

            # ⭐️ 信心度判斷邏輯
            # 如果預測是 OK，但信心度不足，強制轉為 NG (或 unknown)
            # 如果預測本來就是 NG，信心度低也還是 NG
            if conf_score < threshold:
                loginfo("ConvNeXt", f"Low confidence: {conf_score:.4f} < {threshold}. Pred: {pred_class} -> Force NG")
                return "ng" # 信心不足視為 NG
            
        return pred_class
        
    except Exception as e:
        loginfo("ConvNeXt", f"_predict_in_memory failed: {e}")
        return "unknown"


def classify_frame(camera_name, image_bgr, base_dir, date_dir, base_name):
    cfg = CLASSIFY_CFG.get(camera_name)
    if not cfg:
        return None
    model = _MODEL_CACHE.get(cfg["model_path"])
    names = cfg["class_names"]
    if model is None or not names:
        loginfo("ConvNeXt", f"[{camera_name}] Model or class names missing.")
        return None

    out = {"camera": camera_name, "crops": [], "final": None}
    results = []
    
    # ⭐️ 這裡要使用最新的 CONF_THRESHOLD (如果想要支援熱更新，可以在這裡重新 reload config)
    current_thresh = CONF_THRESHOLD 

    for i, off in enumerate(cfg.get("crops", []), 1):
        crop_img = _center_shift_crop(image_bgr, cfg["crop_ratio"], off["dx"], off["dy"])
        
        # ⭐️ MODIFIED: 傳入 current_thresh
        pred = _predict_in_memory(crop_img, model, names, VAL_TRANSFORM, current_thresh)
        
        out["crops"].append({"pred": pred})
        results.append(pred)

    if len(results) == 2 and results[0] == "ok" and results[1] == "ok":
        out["final"] = "ok"
    else:
        out["final"] = "ng"

    return out


def safe_send(sock, addr, val, suffix=".U"):
    """⭐️ MODIFIED: 增加對 sock is None 的檢查"""
    if sock is None:
        print(f"❌ PLC send skipped (socket is None) {addr} <= {val}")
        return
    try:
        sock.Send(addr, val, suffix)
        print(f"📤 PLC {addr} <= {val}")
    except Exception as e:
        print(f"❌ PLC send failed {addr} <= {val}: {e}")
        raise # ⭐️ 拋出異常，讓 camera_task 知道連線已中斷

# =========================
# Connect cameras
# =========================
camera_objects = {}
for name, cfg in cameras.items():
    os.makedirs(cfg["base_dir"], exist_ok=True)
    cam = HuarayCameraController()
    if cam.connect(device_index=cfg["index"]):
        camera_objects[name] = cam
        print(f"✅ Camera {name} CONNECTED (Index {cfg['index']}, Serial {cfg['serial']}) → saving to {cfg['base_dir']}")
        ensure_dirs(cfg["base_dir"])   # create today's dir
    else:
        print(f"❌ Camera {name} FAILED to connect")

# =========================
# Thread loop
# =========================
def camera_task(cam: HuarayCameraController, plc_trigger, plc_result, name, plc_ip, plc_port, base_dir):
    """⭐️ MODIFIED: 
    - 接收 plc_ip, plc_port (不再接收 sock)
    - 內部管理 socket 連線和重連
    - 內部呼叫 cleanup_old_folders
    """
    socket = None
    last_trig = None

    def _connect_plc():
        """(Re)connects the PLC."""
        print(f"[{name}]  attempting to connect PLC {plc_ip}...")
        try:
            sock = plc_socket(plc_ip, plc_port)
            print(f"[{name}] PLC connected.")
            return sock
        except Exception as e:
            print(f"[{name}] PLC connection failed: {e}")
            return None

    # 啟動時進行第一次連線
    socket = _connect_plc()

    while True:
        # ⭐️ 1. 每次迴圈先執行清理
        cleanup_old_folders(base_dir, 5) 
        
        # ⭐️ 2. 檢查 PLC 連線
        if socket is None:
            print(f"[{name}] PLC disconnected. Retrying in 5s...")
            time.sleep(5)
            socket = _connect_plc()
            continue # 進入下一次迴圈

        # ⭐️ 3. 讀取 PLC 狀態 (帶有重連邏輯)
        try:
            raw = socket.Get(plc_trigger, ".D")
            trig = int(raw.strip().splitlines()[0])
        except Exception as e:
            print(f"❌ [{name}] PLC Get error: {e}. Marking for reconnect.")
            socket = None # ⭐️ 標記為斷線，下次迴圈重連
            continue

        # ⭐️ 4. 檢查觸發 (保持不變)
        if trig != last_trig:
            print(f"[{name}] Trigger={trig}{' → Capturing' if trig == 1 else ' → Waiting'}")
            last_trig = trig

        if trig != 1:
            time.sleep(0.5) # ⭐️ 在等待時 sleep，避免空轉
            continue

        # ⭐️ 5. 主邏輯 (帶有重連邏輯)
        try:
            if not cam.start_grabbing():
                raise RuntimeError("start_grabbing() failed")
            time.sleep(0.1)   # small settle time

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{name}_{ts}"

            frame = cam.grab_image_numpy(timeout_ms=3000)
            cam.stop_grabbing()
            if frame is None:
                raise RuntimeError("No frame")

            if frame.ndim == 3 and frame.shape[2] == 3:
                frame = frame[:, :, ::-1]

            date_dir, full_dir = ensure_dirs(base_dir)
            
            # ⭐️ 步驟 1: 先分類
            cls_out = classify_frame(name, frame, base_dir, date_dir, base_name)
            
            if cls_out:
                final = cls_out["final"]
                preds = [c["pred"] for c in cls_out["crops"]]
                print(f"🧠 [{name}] crop_preds={preds} → FINAL={final.upper()}")

                # ⭐️ 步驟 2: 發送最終結果 (1=OK, 3=NG)
                safe_send(socket, plc_result, 1 if final == "ok" else 3, ".U")

                # ⭐️ 步驟 3: 只在 NG 時儲存
                if final == "ng":
                    loginfo("CameraTask", f"[{name}] Result: NG. Saving annotated full frame.")
                    
                    # 儲存帶有標記的完整原圖
                    annotated_full = frame.copy()
                    cv2.putText(annotated_full, "NG", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 15)
                    cv2.putText(annotated_full, name, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 15)
                    
                    full_ng_name = f"{base_name}_NG"
                    ok, full_path = save_full_frame(annotated_full, base_dir, date_dir, full_ng_name)
                    if ok:
                        print(f"📸 [{name}] Saved NG full frame: {full_path}")
                else:
                    loginfo("CameraTask", f"[{name}] Result: OK. No images saved.")
                    
            else:
                print(f"⚠️  [{name}] classification skipped (no config/model)")
                safe_send(socket, plc_result, 3, ".U") # 分類失敗，發送 NG

            # ⭐️ 步驟 4: reset trigger
            time.sleep(0.3)
            safe_send(socket, plc_trigger, 0, ".U")

        except Exception as e:
            # ⭐️ 關鍵：主邏輯（包含 safe_send）出錯
            print(f"❌ [{name}] MainTask ERROR: {traceback.format_exc()}")
            loginfo("CameraTask", f"[{name}] ERROR: {e}")
            try:
                cam.stop_grabbing()
            except Exception:
                pass
            
            # 嘗試發送錯誤訊號
            safe_send(socket, plc_result, 3, ".U") 
            safe_send(socket, plc_trigger, 0, ".U")
            
            socket = None # ⭐️ 確保標記為斷線

# =========================
# Start threads
# =========================
threads = []
for name, cfg in cameras.items():
    cam = camera_objects.get(name)
    if not cam:
        print(f"⏭️  {name} not connected; skipping.")
        continue
    
    # ⭐️ MODIFIED: 更新傳入的參數
    t = threading.Thread(
        target=camera_task,
        args=(
            cam, 
            cfg["plc_trigger"], 
            cfg["plc_result"], 
            name, 
            cfg["plc_ip"],     # ⭐️ NEW
            cfg["plc_port"],   # ⭐️ NEW
            cfg["base_dir"]
        ),
        daemon=True
    )
    t.start()
    threads.append(t)
    print(f"🚀 Started thread for {name}")

for t in threads:
    t.join()
