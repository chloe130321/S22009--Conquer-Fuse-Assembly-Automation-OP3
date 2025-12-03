import os
import cv2
import time
import threading
from datetime import datetime, timedelta
import traceback
import numpy as np

from plc_socket import plc_socket       # ✔ uses Get/Send  :contentReference[oaicite:0]{index=0}
from camera_controller import HuarayCameraController   # ✔  :contentReference[oaicite:1]{index=1}


# ======================================================
# OP3 CAMERA CONFIG
# ======================================================
cameras = {
    "op3_1": {
        "index": 1,
        "serial": "DA26269AAK00007",
        "base_dir": r"C:\3-1_3-3\OP3-1_pictures",
        "plc_trigger": "DM4020",
        "plc_result":  "DM4030",
        "plc_ip": "192.168.162.160",
        "plc_port": 8501,
    },
    "op3_3": {
        "index": 2,
        "serial": "DA26269AAK00010",
        "base_dir": r"C:\3-1_3-3\OP3-3_pictures",
        "plc_trigger": "DM4000",
        "plc_result":  "DM4010",
        "plc_ip": "192.168.162.160",
        "plc_port": 8501,
    },
}


# ======================================================
# Helpers
# ======================================================
def ensure_today_dir(base_dir):
    today = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(base_dir, today)
    os.makedirs(path, exist_ok=True)
    return path


def cleanup_old(base_dir, keep_days=5):
    if not os.path.exists(base_dir):
        return
    cutoff = datetime.now() - timedelta(days=keep_days)
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full):
            continue
        try:
            dt = datetime.strptime(name, "%Y-%m-%d")
            if dt.date() < cutoff.date():
                import shutil
                shutil.rmtree(full)
                print(f"[CLEANUP] removed {full}")
        except:
            continue


def safe_send(sock, addr, val, suffix=".U"):
    """Send DM safely and mark error."""
    if sock is None:
        print(f"❌ PLC send skipped (socket None)  {addr} <= {val}")
        return
    try:
        sock.Send(addr, val, suffix)
        print(f"📤 PLC {addr} <= {val}")
    except Exception as e:
        print(f"❌ PLC send FAILED {addr} <= {val}: {e}")
        raise


# ======================================================
# MAIN THREAD LOOP
# ======================================================
def camera_task(name, cfg, cam_obj):
    plc_ip = cfg["plc_ip"]
    plc_port = cfg["plc_port"]
    dm_trig = cfg["plc_trigger"]
    dm_res  = cfg["plc_result"]
    base_dir = cfg["base_dir"]

    socket = None
    last_val = None

    def connect_plc():
        print(f"[{name}] connecting PLC {plc_ip}:{plc_port} ...")
        try:
            sock = plc_socket(plc_ip, plc_port)
            print(f"[{name}] PLC CONNECTED")
            return sock
        except Exception as e:
            print(f"[{name}] PLC connect failed: {e}")
            return None

    socket = connect_plc()

    while True:
        cleanup_old(base_dir, 5)

        # reconnect if socket lost
        if socket is None:
            print(f"[{name}] retry PLC in 5s…")
            time.sleep(5)
            socket = connect_plc()
            continue

        # read trigger
        try:
            raw = socket.Get(dm_trig, ".D")
            val = int(raw.strip().splitlines()[0])
        except Exception as e:
            print(f"[{name}] PLC read error: {e} → reconnect")
            socket = None
            continue

        if val != last_val:
            print(f"[{name}] Trigger={val}")
            last_val = val

        # wait until trigger = 1
        if val != 1:
            time.sleep(0.2)
            continue

        # triggered → capture
        try:
            print(f"[{name}] Capturing...")

            if not cam_obj.start_grabbing():
                raise RuntimeError("start_grabbing() failed")

            time.sleep(0.1)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fname = f"{name}_{ts}.png"

            frame = cam_obj.grab_image_numpy(timeout_ms=3000)
            cam_obj.stop_grabbing()

            if frame is None:
                raise RuntimeError("No frame captured")

            # convert grayscale to BGR
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            today_dir = ensure_today_dir(base_dir)
            save_path = os.path.join(today_dir, fname)
            cv2.imwrite(save_path, frame)
            print(f"📸 Saved: {save_path}")

            # send OK result
            safe_send(socket, dm_res, 1, ".U")

        except Exception as e:
            print(f"❌ [{name}] ERROR:\n{traceback.format_exc()}")
            try:
                cam_obj.stop_grabbing()
            except:
                pass
            safe_send(socket, dm_res, 3, ".U")
            socket = None

        # reset trigger
        time.sleep(0.3)
        safe_send(socket, dm_trig, 0, ".U")


# ======================================================
# STARTUP
# ======================================================
if __name__ == "__main__":
    camera_objects = {}

    # connect cameras
    for name, cfg in cameras.items():
        os.makedirs(cfg["base_dir"], exist_ok=True)

        cam = HuarayCameraController()
        if cam.connect(device_index=cfg["index"]):
            print(f"✅ Camera {name} connected")
            ensure_today_dir(cfg["base_dir"])
            camera_objects[name] = cam
        else:
            print(f"❌ Camera {name} FAILED")
    
    # start threads
    threads = []
    for name, cfg in cameras.items():
        if name not in camera_objects:
            print(f"⏭️ {name} skipped (no camera)")
            continue

        t = threading.Thread(
            target=camera_task,
            args=(name, cfg, camera_objects[name]),
            daemon=True
        )
        t.start()
        threads.append(t)
        print(f"🚀 Thread started for {name}")

    for t in threads:
        t.join()
