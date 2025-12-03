# src/controllers/camera_sdk/camera_controller.py

import os
import sys
import ctypes
from contextlib import contextmanager
import cv2
import numpy as np
import tkinter as tk

try:
    from .IMVApi import MvCamera
    from .IMVDefines import *
except:
    from IMVApi import MvCamera
    from IMVDefines import *

class HuarayCameraController:
    """
    用於華睿相機的高階控制器。
    封裝了初始化、連線、取像和斷開連線的複雜邏輯。
    """
    def __init__(self):
        try:
            self.cam = MvCamera()
        except RuntimeError as e:
            print(f"致命錯誤：無法建立相機底層物件。錯誤訊息: {e}")
            raise

        self.is_connected = False
        self.is_grabbing = False

    def connect(self, device_index: int = 0) -> bool:
        if self.is_connected:
            print("相機已經連線。")
            return True

        print("正在尋找相機...")
        device_list = IMV_DeviceList()
        status = self.cam.IMV_EnumDevices(device_list, IMV_EInterfaceType.interfaceTypeAll)
        
        if status != IMV_OK or device_list.nDevNum == 0:
            print("錯誤：找不到任何相機設備。")
            return False

        if device_index >= device_list.nDevNum:
            print(f"錯誤：指定的設備索引 {device_index} 超出範圍 (共 {device_list.nDevNum} 台)。")
            return False

        print(f"找到 {device_list.nDevNum} 台相機，正在連接第 {device_index} 台...")
        status = self.cam.IMV_CreateHandle(IMV_ECreateHandleMode.modeByIndex, byref(c_int(device_index)))
        if status != IMV_OK:
            print(f"錯誤：建立相機句柄失敗，狀態碼: {status}")
            return False
        
        status = self.cam.IMV_Open()
        if status != IMV_OK:
            print(f"錯誤：開啟相機失敗，狀態碼: {status}")
            self.cam.IMV_DestroyHandle()
            return False
        
        self.is_connected = True
        print("相機連線成功！")
        return True

    def disconnect(self):
        if not self.is_connected:
            return
            
        print("正在斷開相機連線...")
        if self.is_grabbing:
            self.stop_grabbing()
        
        self.cam.IMV_Close()
        self.cam.IMV_DestroyHandle()
        
        self.is_connected = False
        self.is_grabbing = False
        print("相機已斷開。")
        
    def start_grabbing(self) -> bool:
        if not self.is_connected:
            print("錯誤: 相機未連線。")
            return False
        if self.is_grabbing:
            return True
        
        status = self.cam.IMV_StartGrabbing()
        if status == IMV_OK:
            self.is_grabbing = True
            print("相機已開始取像。")
            return True
        else:
            print(f"錯誤：開始取像失敗，狀態碼: {status}")
            return False

    def stop_grabbing(self):
        if not self.is_grabbing:
            return
        self.cam.IMV_StopGrabbing()
        self.is_grabbing = False
        print("相機已停止取像。")

    def get_frame(self, timeout_ms: int = 2000) -> IMV_Frame | None:
        if not self.is_grabbing:
            print("錯誤：未開始取像，無法取得影像。")
            return None
        frame = IMV_Frame()
        status = self.cam.IMV_GetFrame(frame, timeout_ms)
        if status == IMV_OK:
            return frame
        else:
            if status != -114: # -114 是超時，為正常現象，不顯示
                 print(f"錯誤：取得影像失敗，狀態碼: {status}")
            return None
            
    def release_frame(self, frame: IMV_Frame):
        if frame:
            self.cam.IMV_ReleaseFrame(frame)

    # --- 新增：高階參數控制方法 (已修正) ---
    def get_exposure(self) -> float | None:
        if not self.is_connected: return None
        exposure_time = c_double()
        # 修正：直接傳遞 ctypes 物件，而不是 byref() 的結果
        if self.cam.IMV_GetDoubleFeatureValue("ExposureTime", exposure_time) == IMV_OK:
            return exposure_time.value
        return None

    def set_exposure(self, value: float) -> bool:
        if not self.is_connected: return False
        return self.cam.IMV_SetDoubleFeatureValue("ExposureTime", value) == IMV_OK

    def get_gain(self) -> float | None:
        if not self.is_connected: return None
        gain = c_double()
        # 修正：直接傳遞 ctypes 物件，而不是 byref() 的結果
        if self.cam.IMV_GetDoubleFeatureValue("Gain", gain) == IMV_OK:
            return gain.value
        return None

    def set_gain(self, value: float) -> bool:
        if not self.is_connected: return False
        return self.cam.IMV_SetDoubleFeatureValue("Gain", value) == IMV_OK
    
    def _convert_frame_to_numpy(self, frame: IMV_Frame):
        """
        [這是一個新的內部方法]
        將相機的 _IMV_Frame_ 物件轉換為 OpenCV 的 NumPy 陣列。
        這段邏輯是從 AutomationManager 移入，使其成為相機控制器的核心功能。
        """
        if not frame or frame.frameInfo.size == 0:
            return None
        
        info = frame.frameInfo
        p_data_as_char = ctypes.cast(frame.pData, ctypes.POINTER(ctypes.c_ubyte * info.size))
        image_1d = np.frombuffer(p_data_as_char.contents, dtype=np.uint8)
        pixel_format = info.pixelFormat
        
        # 根據像素格式進行轉換
        if pixel_format == IMV_EPixelType.gvspPixelMono8:
            return image_1d.reshape((info.height, info.width))
        elif pixel_format in [IMV_EPixelType.gvspPixelBGR8, IMV_EPixelType.gvspPixelRGB8]:
            image_bgr = image_1d.reshape((info.height, info.width, 3))
            if pixel_format == IMV_EPixelType.gvspPixelRGB8:
                return cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)
            return image_bgr
        elif pixel_format in [IMV_EPixelType.gvspPixelBayRG8, IMV_EPixelType.gvspPixelBayGB8,
                              IMV_EPixelType.gvspPixelBayGR8, IMV_EPixelType.gvspPixelBayBG8]:
            bayer_image = image_1d.reshape((info.height, info.width))
            convert_map = {
                IMV_EPixelType.gvspPixelBayRG8: cv2.COLOR_BAYER_RG2BGR,
                IMV_EPixelType.gvspPixelBayGB8: cv2.COLOR_BAYER_GB2BGR,
                IMV_EPixelType.gvspPixelBayGR8: cv2.COLOR_BAYER_GR2BGR,
                IMV_EPixelType.gvspPixelBayBG8: cv2.COLOR_BAYER_BG2BGR
            }
            return cv2.cvtColor(bayer_image, convert_map[pixel_format])
        else:
            print(f"警告：遇到未處理的像素格式 {pixel_format}")
            return None

    def grab_image_numpy(self, timeout_ms=2000):
        if not self.is_grabbing:
            print("錯誤: 相機未處於取像狀態 (is_grabbing=False)。")
            return None

        frame_object = None
        try:
            # 1. 從相機取得原始 Frame 物件
            frame_object = self.get_frame(timeout_ms)
            if frame_object:
                # 2. 將 Frame 轉換為 NumPy 陣列
                numpy_image = self._convert_frame_to_numpy(frame_object)
                return numpy_image
            else:
                # 如果 get_frame 超時或失敗，會返回 None
                return None
        finally:
            # 3. 【關鍵】無論成功與否，都必須釋放 Frame 物件
            if frame_object:
                self.release_frame(frame_object)


# ... (檔案下方的獨立測試區塊維持不變) ...
# --- 獨立測試區塊 ---
def _get_screen_resolution():
    try:
        root = tk.Tk()
        root.withdraw()
        width, height = root.winfo_screenwidth(), root.winfo_screenheight()
        return width, height
    except Exception:
        return 1920, 1080

def _convert_frame_to_image(frame: IMV_Frame):
    if not frame or frame.frameInfo.size == 0: return None
    info = frame.frameInfo
    p_data_as_char = ctypes.cast(frame.pData, ctypes.POINTER(ctypes.c_ubyte * info.size))
    image_1d = np.frombuffer(p_data_as_char.contents, dtype=np.uint8)
    pixel_format = info.pixelFormat
    if pixel_format == IMV_EPixelType.gvspPixelMono8:
        return image_1d.reshape((info.height, info.width))
    elif pixel_format in [IMV_EPixelType.gvspPixelBGR8, IMV_EPixelType.gvspPixelRGB8]:
        image_bgr = image_1d.reshape((info.height, info.width, 3))
        if pixel_format == IMV_EPixelType.gvspPixelRGB8:
            return cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)
        return image_bgr
    elif pixel_format in [IMV_EPixelType.gvspPixelBayRG8, IMV_EPixelType.gvspPixelBayGB8,
                          IMV_EPixelType.gvspPixelBayGR8, IMV_EPixelType.gvspPixelBayBG8]:
        bayer_image = image_1d.reshape((info.height, info.width))
        if pixel_format == IMV_EPixelType.gvspPixelBayRG8: convert_code = cv2.COLOR_BAYER_RG2RGB
        elif pixel_format == IMV_EPixelType.gvspPixelBayGB8: convert_code = cv2.COLOR_BAYER_GB2BGR
        elif pixel_format == IMV_EPixelType.gvspPixelBayGR8: convert_code = cv2.COLOR_BAYER_GR2BGR
        else: convert_code = cv2.COLOR_BAYER_BG2BGR
        return cv2.cvtColor(bayer_image, convert_code)
    return None

def _test():
    print("--- 開始獨立測試 HuarayCameraController (含即時預覽) ---")
    camera = None
    window_name = "Camera Live Preview"
    try:
        camera = HuarayCameraController()
        if camera.connect():
            if camera.start_grabbing():
                screen_w, screen_h = _get_screen_resolution()
                max_win_h = int(screen_h * 0.8)
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, int(max_win_h * 1.5), max_win_h)
                print("\n相機預覽已開啟。按 'q' 或關閉視窗來結束程式。")
                while True:
                    frame = camera.get_frame()
                    if frame:
                        image = _convert_frame_to_image(frame)
                        if image is not None:
                            cv2.imshow(window_name, image)
                        camera.release_frame(frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("偵測到 'q' 鍵，正在關閉...")
                        break
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        print("偵測到視窗已關閉，正在結束...")
                        break
    except RuntimeError:
        print("HuarayCameraController 初始化失敗，測試終止。")
    except Exception as e:
        print(f"測試過程中發生未預期錯誤: {e}")
    finally:
        if camera and camera.is_connected:
            camera.disconnect()
        cv2.destroyAllWindows()
    print("--- HuarayCameraController 獨立測試結束 ---\n")

if __name__ == "__main__":
    _test()