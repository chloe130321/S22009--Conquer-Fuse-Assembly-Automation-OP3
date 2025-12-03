# camera_sdk/IMVApi.py (完整修正版)

import sys
import ctypes
import platform
import os
try:
    # 當作為套件被導入時，此行會成功
    from .IMVDefines import *
except ImportError:
    # 當直接執行時，上一行會失敗，轉而執行此行
    from IMVDefines import *

# 建立一個全域變數來緩存 DLL 物件，避免重複載入
g_sdk_dll_obj = None

def load_sdk_dll():
    """
    載入並返回 SDK DLL 物件。
    使用全域變數 g_sdk_dll_obj 來確保整個程式只載入一次。
    """
    global g_sdk_dll_obj
    if g_sdk_dll_obj:
        return g_sdk_dll_obj

    if sys.platform != 'win32':
        print("錯誤：此範例目前僅支援 Windows 平台。")
        return None

    # --- 關鍵：使用 __file__ 來計算相對於本檔案的路徑 ---
    # 這樣無論從哪裡呼叫，路徑都是正確的
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bits, _ = platform.architecture()
    
    if bits == '64bit':
        dll_path = os.path.join(script_dir, "IMV", "Runtime", "x64")
    else:
        dll_path = os.path.join(script_dir, "IMV", "Runtime", "Win32")
    
    dll_file = os.path.join(dll_path, "MVSDKmd.dll")

    if not os.path.exists(dll_file):
        print(f"致命錯誤：在指定路徑下找不到 DLL 檔案: {dll_file}")
        return None

    original_cwd = os.getcwd()
    try:
        # 切換工作目錄來讓 DLL 找到其相依性檔案
        os.chdir(dll_path)
        os.add_dll_directory(dll_path)
        g_sdk_dll_obj = ctypes.WinDLL(dll_file)
        print("SDK Library loaded successfully!")
        return g_sdk_dll_obj
    except Exception as e:
        print(f"致命錯誤：載入 SDK DLL 時發生例外: {e}")
        g_sdk_dll_obj = None
        return None
    finally:
        # 無論成功或失敗，都必須切換回原本的工作目錄
        os.chdir(original_cwd)


class MvCamera():
    """
    華睿相機 SDK 的低階 API 封裝。
    所有方法都透過類別屬性 sdk_dll 來呼叫 C 函式庫。
    """
    
    # 在定義類別時，就執行一次載入函式，並將結果存為類別屬性
    sdk_dll = load_sdk_dll()

    def __init__(self):
        # 在建立物件時，檢查 DLL 是否已成功載入
        if not MvCamera.sdk_dll:
            raise RuntimeError("MvCamera 初始化失敗：SDK DLL 未能載入。請檢查上述錯誤訊息。")
            
        self._handle = c_void_p()
        self.handle = pointer(self._handle)

    # --- 以下所有方法都從呼叫 MvCamera.sdk_dll 改為呼叫 MvCamera.sdk_dll ---

    @staticmethod
    def IMV_GetVersion():
        MvCamera.sdk_dll.IMV_GetVersion.restype = c_char_p
        return MvCamera.sdk_dll.IMV_GetVersion()

    @staticmethod
    def IMV_EnumDevices(pDeviceList, interfaceType):
        MvCamera.sdk_dll.IMV_EnumDevices.argtype = (c_void_p, c_uint)
        MvCamera.sdk_dll.IMV_EnumDevices.restype = c_int
        return MvCamera.sdk_dll.IMV_EnumDevices(byref(pDeviceList), c_uint(interfaceType))

    @staticmethod
    def IMV_EnumDevicesByUnicast(pDeviceList, pIpAddress):
        MvCamera.sdk_dll.IMV_EnumDevicesByUnicast.argtype = (c_void_p, c_char_p)
        MvCamera.sdk_dll.IMV_EnumDevicesByUnicast.restype = c_int
        return MvCamera.sdk_dll.IMV_EnumDevicesByUnicast(byref(pDeviceList), pIpAddress.encode('ascii'))

    def IMV_CreateHandle(self, mode, pIdentifier):
        MvCamera.sdk_dll.IMV_CreateHandle.argtype = (c_void_p, c_int, c_void_p)
        MvCamera.sdk_dll.IMV_CreateHandle.restype = c_int
        return MvCamera.sdk_dll.IMV_CreateHandle(byref(self.handle), c_int(mode), pIdentifier)

    def IMV_DestroyHandle(self):
        MvCamera.sdk_dll.IMV_DestroyHandle.argtype = c_void_p
        MvCamera.sdk_dll.IMV_DestroyHandle.restype = c_int
        return MvCamera.sdk_dll.IMV_DestroyHandle(self.handle)

    def IMV_GetDeviceInfo(self, pDevInfo):
        MvCamera.sdk_dll.IMV_GetDeviceInfo.argtype = (c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_GetDeviceInfo.restype = c_int
        return MvCamera.sdk_dll.IMV_GetDeviceInfo(self.handle, byref(pDevInfo))

    def IMV_Open(self):
        MvCamera.sdk_dll.IMV_Open.argtype = c_void_p
        MvCamera.sdk_dll.IMV_Open.restype = c_int
        return MvCamera.sdk_dll.IMV_Open(self.handle)

    def IMV_OpenEx(self, accessPermission):
        MvCamera.sdk_dll.IMV_OpenEx.argtype = (c_void_p, c_int)
        MvCamera.sdk_dll.IMV_OpenEx.restype = c_int
        return MvCamera.sdk_dll.IMV_OpenEx(self.handle, c_int(accessPermission))

    def IMV_IsOpen(self):
        MvCamera.sdk_dll.IMV_IsOpen.argtype = c_void_p
        MvCamera.sdk_dll.IMV_IsOpen.restype = c_bool
        return MvCamera.sdk_dll.IMV_IsOpen(self.handle)

    def IMV_Close(self):
        MvCamera.sdk_dll.IMV_Close.argtype = c_void_p
        MvCamera.sdk_dll.IMV_Close.restype = c_int
        return MvCamera.sdk_dll.IMV_Close(self.handle)
    
    def IMV_IsGrabbing(self):
        MvCamera.sdk_dll.IMV_IsGrabbing.argtype = c_void_p
        MvCamera.sdk_dll.IMV_IsGrabbing.restype = c_bool
        return MvCamera.sdk_dll.IMV_IsGrabbing(self.handle)

    def IMV_StartGrabbing(self):
        MvCamera.sdk_dll.IMV_StartGrabbing.argtype = c_void_p
        MvCamera.sdk_dll.IMV_StartGrabbing.restype = c_int
        return MvCamera.sdk_dll.IMV_StartGrabbing(self.handle)

    def IMV_StopGrabbing(self):
        MvCamera.sdk_dll.IMV_StopGrabbing.argtype = c_void_p
        MvCamera.sdk_dll.IMV_StopGrabbing.restype = c_int
        return MvCamera.sdk_dll.IMV_StopGrabbing(self.handle)

    def IMV_GetFrame(self, pFrame, timeoutMS):
        MvCamera.sdk_dll.IMV_GetFrame.argtype = (c_void_p, c_void_p, c_uint)
        MvCamera.sdk_dll.IMV_GetFrame.restype = c_int
        return MvCamera.sdk_dll.IMV_GetFrame(self.handle, byref(pFrame), c_uint(timeoutMS))

    def IMV_ReleaseFrame(self, pFrame):
        MvCamera.sdk_dll.IMV_ReleaseFrame.argtype = (c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_ReleaseFrame.restype = c_int
        return MvCamera.sdk_dll.IMV_ReleaseFrame(self.handle, byref(pFrame))

    # ch:释放图像缓存
    def IMV_CloneFrame(self, pFrame, pCloneFrame):
        MvCamera.sdk_dll.IMV_CloneFrame.argtype = (c_void_p, c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_CloneFrame.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_CloneFrame(IN IMV_HANDLE handle, IN IMV_Frame* pFrame, OUT IMV_Frame* pCloneFrame);
        return MvCamera.sdk_dll.IMV_CloneFrame(self.handle, byref(pFrame), byref(pCloneFrame))

    # ch:获取Chunk数据(仅对GigE/Usb相机有效)
    def IMV_GetChunkDataByIndex(self, pFrame, index, pChunkDataInfo):
        MvCamera.sdk_dll.IMV_GetChunkDataByIndex.argtype = (c_void_p, c_void_p, c_uint, c_void_p)
        MvCamera.sdk_dll.IMV_GetChunkDataByIndex.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_GetChunkDataByIndex(IN IMV_HANDLE handle, IN IMV_Frame* pFrame, IN unsigned int index, OUT IMV_ChunkDataInfo *pChunkDataInfo);
        return MvCamera.sdk_dll.IMV_GetChunkDataByIndex(self.handle, byref(pFrame), c_uint(index), byref(pChunkDataInfo))

    # ch:获取流统计信息(IMV_StartGrabbing / IMV_StartGrabbing执行后调用)
    def IMV_GetStatisticsInfo(self, pStreamStatsInfo):
        MvCamera.sdk_dll.IMV_GetStatisticsInfo.argtype = (c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_GetStatisticsInfo.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_GetStatisticsInfo(IN IMV_HANDLE handle, OUT IMV_StreamStatisticsInfo* pStreamStatsInfo);
        return MvCamera.sdk_dll.IMV_GetStatisticsInfo(self.handle, byref(pStreamStatsInfo))

    # ch:重置流统计信息(IMV_StartGrabbing / IMV_StartGrabbing执行后调用)
    def IMV_ResetStatisticsInfo(self):
        MvCamera.sdk_dll.IMV_ResetStatisticsInfo.argtype = c_void_p
        MvCamera.sdk_dll.IMV_ResetStatisticsInfo.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_ResetStatisticsInfo(IN IMV_HANDLE handle);
        return MvCamera.sdk_dll.IMV_ResetStatisticsInfo(self.handle)

    # ch:判断属性是否可用
    def IMV_FeatureIsAvailable(self, pFeatureName):
        MvCamera.sdk_dll.IMV_FeatureIsAvailable.argtype = (c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_FeatureIsAvailable.restype = c_bool
        # C原型:IMV_API bool IMV_CALL IMV_FeatureIsAvailable(IN IMV_HANDLE handle, IN const char* pFeatureName);
        return MvCamera.sdk_dll.IMV_FeatureIsAvailable(self.handle, pFeatureName.encode('ascii'))

    # ch:判断属性是否可读
    def IMV_FeatureIsReadable(self, pFeatureName):
        MvCamera.sdk_dll.IMV_FeatureIsReadable.argtype = (c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_FeatureIsReadable.restype = c_bool
        # C原型:IMV_API bool IMV_CALL IMV_FeatureIsReadable(IN IMV_HANDLE handle, IN const char* pFeatureName);
        return MvCamera.sdk_dll.IMV_FeatureIsReadable(self.handle, pFeatureName.encode('ascii'))

    # ch:判断属性是否可写
    def IMV_FeatureIsWriteable(self, pFeatureName):
        MvCamera.sdk_dll.IMV_FeatureIsWriteable.argtype = (c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_FeatureIsWriteable.restype = c_bool
        # C原型:IMV_API bool IMV_CALL IMV_FeatureIsWriteable(IN IMV_HANDLE handle, IN const char* pFeatureName);
        return MvCamera.sdk_dll.IMV_FeatureIsWriteable(self.handle, pFeatureName.encode('ascii'))

    # ch:判断属性是否可流
    def IMV_FeatureIsStreamable(self, pFeatureName):
        MvCamera.sdk_dll.IMV_FeatureIsStreamable.argtype = (c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_FeatureIsStreamable.restype = c_bool
        # C原型:IMV_API bool IMV_CALL IMV_FeatureIsStreamable(IN IMV_HANDLE handle, IN const char* pFeatureName);
        return MvCamera.sdk_dll.IMV_FeatureIsStreamable(self.handle, pFeatureName.encode('ascii'))

    # ch:判断属性是否有效
    def IMV_FeatureIsValid(self, pFeatureName):
        MvCamera.sdk_dll.IMV_FeatureIsValid.argtype = (c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_FeatureIsValid.restype = c_bool
        # C原型:IMV_API bool IMV_CALL IMV_FeatureIsValid(IN IMV_HANDLE handle, IN const char* pFeatureName);
        return MvCamera.sdk_dll.IMV_FeatureIsValid(self.handle, pFeatureName.encode('ascii'))

    # ch:获取属性类型
    def IMV_GetFeatureType(self, pFeatureName, pPropertyType):
        MvCamera.sdk_dll.IMV_GetFeatureType.argtype = (c_void_p, c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_GetFeatureType.restype = c_bool
        # C原型:IMV_API bool IMV_CALL IMV_GetFeatureType(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT IMV_EFeatureType* pPropertyType);
        return MvCamera.sdk_dll.IMV_GetFeatureType(self.handle, pFeatureName.encode('ascii'), byref(pPropertyType))

    # ch:获取整型属性值
    def IMV_GetIntFeatureValue(self, pFeatureName, pIntValue):
        MvCamera.sdk_dll.IMV_GetIntFeatureValue.argtype = (c_void_p, c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_GetIntFeatureValue.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_GetIntFeatureValue(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT int64_t* pIntValue);
        return MvCamera.sdk_dll.IMV_GetIntFeatureValue(self.handle, pFeatureName.encode('ascii'), byref(pIntValue))

    # ch:获取整型属性可设的最小值
    def IMV_GetIntFeatureMin(self, pFeatureName, pIntValue):
        MvCamera.sdk_dll.IMV_GetIntFeatureMin.argtype = (c_void_p, c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_GetIntFeatureMin.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_GetIntFeatureMin(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT int64_t* pIntValue);
        return MvCamera.sdk_dll.IMV_GetIntFeatureMin(self.handle, pFeatureName.encode('ascii'), byref(pIntValue))

    # ch:获取整型属性可设的最大值
    def IMV_GetIntFeatureMax(self, pFeatureName, pIntValue):
        MvCamera.sdk_dll.IMV_GetIntFeatureMax.argtype = (c_void_p, c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_GetIntFeatureMax.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_GetIntFeatureMax(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT int64_t* pIntValue);
        return MvCamera.sdk_dll.IMV_GetIntFeatureMax(self.handle, pFeatureName.encode('ascii'), byref(pIntValue))

    # ch:获取整型属性步长
    def IMV_GetIntFeatureInc(self, pFeatureName, pIntValue):
        MvCamera.sdk_dll.IMV_GetIntFeatureInc.argtype = (c_void_p, c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_GetIntFeatureInc.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_GetIntFeatureInc(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT int64_t* pIntValue);
        return MvCamera.sdk_dll.IMV_GetIntFeatureInc(self.handle, pFeatureName.encode('ascii'), byref(pIntValue))

    # ch:设置整型属性值
    def IMV_SetIntFeatureValue(self, pFeatureName, pIntValue):
        MvCamera.sdk_dll.IMV_SetIntFeatureValue.argtype = (c_void_p, c_void_p, c_int)
        MvCamera.sdk_dll.IMV_SetIntFeatureValue.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_SetIntFeatureValue(IN IMV_HANDLE handle, IN const char* pFeatureName, IN int64_t intValue);
        return MvCamera.sdk_dll.IMV_SetIntFeatureValue(self.handle, pFeatureName.encode('ascii'), pIntValue)

    # ch:获取浮点属性值
    def IMV_GetDoubleFeatureValue(self, pFeatureName, pDoubleValue):
        MvCamera.sdk_dll.IMV_GetDoubleFeatureValue.argtype = (c_void_p, c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_GetDoubleFeatureValue.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_GetDoubleFeatureValue(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT double* pDoubleValue);
        return MvCamera.sdk_dll.IMV_GetDoubleFeatureValue(self.handle, pFeatureName.encode('ascii'), byref(pDoubleValue))

    # ch:获取浮点属性可设的最小值
    def IMV_GetDoubleFeatureMin(self, pFeatureName, pDoubleValue):
        MvCamera.sdk_dll.IMV_GetDoubleFeatureMin.argtype = (c_void_p, c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_GetDoubleFeatureMin.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_GetDoubleFeatureMin(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT double* pDoubleValue);
        return MvCamera.sdk_dll.IMV_GetDoubleFeatureMin(self.handle, pFeatureName.encode('ascii'), byref(pDoubleValue))

    # ch:获取浮点属性可设的最大值
    def IMV_GetDoubleFeatureMax(self, pFeatureName, pDoubleValue):
        MvCamera.sdk_dll.IMV_GetDoubleFeatureMax.argtype = (c_void_p, c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_GetDoubleFeatureMax.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_GetDoubleFeatureMax(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT double* pDoubleValue);
        return MvCamera.sdk_dll.IMV_GetDoubleFeatureMax(self.handle, pFeatureName.encode('ascii'), byref(pDoubleValue))

    # ch:设置浮点属性值
    def IMV_SetDoubleFeatureValue(self, pFeatureName, doubleValue):
        MvCamera.sdk_dll.IMV_SetDoubleFeatureValue.argtype = (c_void_p, c_void_p, c_double)
        MvCamera.sdk_dll.IMV_SetDoubleFeatureValue.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_SetDoubleFeatureValue(IN IMV_HANDLE handle, IN const char* pFeatureName, IN double doubleValue);
        return MvCamera.sdk_dll.IMV_SetDoubleFeatureValue(self.handle, pFeatureName.encode('ascii'), c_double(doubleValue))

    # ch:获取布尔属性值
    def IMV_GetBoolFeatureValue(self, pFeatureName, pBoolValue):
        MvCamera.sdk_dll.IMV_GetBoolFeatureValue.argtype = (c_void_p, c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_GetBoolFeatureValue.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_GetBoolFeatureValue(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT bool* pBoolValue);
        return MvCamera.sdk_dll.IMV_GetBoolFeatureValue(self.handle, pFeatureName.encode('ascii'), byref(pBoolValue))

    # ch:设置布尔属性值
    def IMV_SetBoolFeatureValue(self, pFeatureName, boolValue):
        MvCamera.sdk_dll.IMV_SetBoolFeatureValue.argtype = (c_void_p, c_void_p, c_bool)
        MvCamera.sdk_dll.IMV_SetBoolFeatureValue.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_SetBoolFeatureValue(IN IMV_HANDLE handle, IN const char* pFeatureName, IN bool boolValue);
        return MvCamera.sdk_dll.IMV_SetBoolFeatureValue(self.handle, pFeatureName.encode('ascii'), c_bool(boolValue))

    # ch:获取枚举属性值
    def IMV_GetEnumFeatureValue(self, pFeatureName, pEnumValue):
        MvCamera.sdk_dll.IMV_GetEnumFeatureValue.argtype = (c_void_p, c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_GetEnumFeatureValue.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_GetEnumFeatureValue(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT uint64_t* pEnumValue);
        return MvCamera.sdk_dll.IMV_GetEnumFeatureValue(self.handle, pFeatureName.encode('ascii'), byref(pEnumValue))

    # ch:设置枚举属性值
    def IMV_SetEnumFeatureValue(self, pFeatureName, enumValue):
        MvCamera.sdk_dll.IMV_SetEnumFeatureValue.argtype = (c_void_p, c_void_p, c_uint64)
        MvCamera.sdk_dll.IMV_SetEnumFeatureValue.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_SetEnumFeatureValue(IN IMV_HANDLE handle, IN const char* pFeatureName, IN uint64_t enumValue);
        return MvCamera.sdk_dll.IMV_SetEnumFeatureValue(self.handle, pFeatureName.encode('ascii'), c_int64(enumValue))

    # ch:获取枚举属性symbol值
    def IMV_GetEnumFeatureSymbol(self, pFeatureName, pEnumSymbol):
        MvCamera.sdk_dll.IMV_GetEnumFeatureSymbol.argtype = (c_void_p, c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_GetEnumFeatureSymbol.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_GetEnumFeatureSymbol(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT IMV_String* pEnumSymbol);
        return MvCamera.sdk_dll.IMV_GetEnumFeatureSymbol(self.handle, pFeatureName.encode('ascii'), byref(pEnumSymbol))

    # ch:设置枚举属性symbol值
    def IMV_SetEnumFeatureSymbol(self, pFeatureName, pEnumSymbol):
        MvCamera.sdk_dll.IMV_SetEnumFeatureSymbol.argtype = (c_void_p, c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_SetEnumFeatureSymbol.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_SetEnumFeatureSymbol(IN IMV_HANDLE handle, IN const char* pFeatureName, IN const char* pEnumSymbol);
        return MvCamera.sdk_dll.IMV_SetEnumFeatureSymbol(self.handle, pFeatureName.encode('ascii'), pEnumSymbol.encode('ascii'))

    # ch:获取枚举属性的可设枚举值的个数
    def IMV_GetEnumFeatureEntryNum(self, pFeatureName, pEntryNum):
        MvCamera.sdk_dll.IMV_GetEnumFeatureEntryNum.argtype = (c_void_p, c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_GetEnumFeatureEntryNum.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_GetEnumFeatureEntryNum(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT unsigned int* pEntryNum);
        return MvCamera.sdk_dll.IMV_GetEnumFeatureEntryNum(self.handle, pFeatureName.encode('ascii'), byref(pEntryNum))

    # ch:获取枚举属性的可设枚举值列表
    def IMV_GetEnumFeatureEntrys(self, pFeatureName, pEnumEntryList):
        MvCamera.sdk_dll.IMV_GetEnumFeatureEntrys.argtype = (c_void_p, c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_GetEnumFeatureEntrys.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_GetEnumFeatureEntrys(IN IMV_HANDLE handle, IN const char* pFeatureName, IN_OUT IMV_EnumEntryList* pEnumEntryList);
        return MvCamera.sdk_dll.IMV_GetEnumFeatureEntrys(self.handle, pFeatureName.encode('ascii'), byref(pEnumEntryList))

    # ch:获取字符串属性值
    def IMV_GetStringFeatureValue(self, pFeatureName, pStringValue):
        MvCamera.sdk_dll.IMV_GetStringFeatureValue.argtype = (c_void_p, c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_GetStringFeatureValue.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_GetStringFeatureValue(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT IMV_String* pStringValue);
        return MvCamera.sdk_dll.IMV_GetStringFeatureValue(self.handle, pFeatureName.encode('ascii'), byref(pStringValue))

    # ch:设置字符串属性值
    def IMV_SetStringFeatureValue(self, pFeatureName, pStringValue):
        MvCamera.sdk_dll.IMV_SetStringFeatureValue.argtype = (c_void_p, c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_SetStringFeatureValue.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_SetStringFeatureValue(IN IMV_HANDLE handle, IN const char* pFeatureName, IN const char* pStringValue);
        return MvCamera.sdk_dll.IMV_SetStringFeatureValue(self.handle, pFeatureName.encode('ascii'),
                                                  pStringValue.encode('ascii'))

    # ch:执行命令属性
    def IMV_ExecuteCommandFeature(self, pFeatureName):
        MvCamera.sdk_dll.IMV_ExecuteCommandFeature.argtype = (c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_ExecuteCommandFeature.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_ExecuteCommandFeature(IN IMV_HANDLE handle, IN const char* pFeatureName);
        return MvCamera.sdk_dll.IMV_ExecuteCommandFeature(self.handle, pFeatureName.encode('ascii'))

    # ch:像素格式转换
    def IMV_PixelConvert(self, pstPixelConvertParam):
        MvCamera.sdk_dll.IMV_PixelConvert.argtype = (c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_PixelConvert.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_PixelConvert(IN IMV_HANDLE handle, IN_OUT IMV_PixelConvertParam* pstPixelConvertParam);
        return MvCamera.sdk_dll.IMV_PixelConvert(self.handle, byref(pstPixelConvertParam))

    # ch:图像翻转
    def IMV_FlipImage(self, pstFlipImageParam):
        MvCamera.sdk_dll.IMV_FlipImage.argtype = (c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_FlipImage.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_FlipImage(IN IMV_HANDLE handle, IN_OUT IMV_FlipImageParam* pstFlipImageParam);
        return MvCamera.sdk_dll.IMV_FlipImage(self.handle, byref(pstFlipImageParam))

    # ch:图像顺时针旋转
    def IMV_RotateImage(self, pstRotateImageParam):
        MvCamera.sdk_dll.IMV_RotateImage.argtype = (c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_RotateImage.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_RotateImage(IN IMV_HANDLE handle, IN_OUT IMV_RotateImageParam* pstRotateImageParam);
        return MvCamera.sdk_dll.IMV_RotateImage(self.handle, byref(pstRotateImageParam))


    # ch:图像顺时针旋转
    def IMV_InternalWriteReg(self, regAddress, regValue, pLength):
        MvCamera.sdk_dll.IMV_InternalWriteReg.argtype = (c_void_p, c_uint64, c_uint64, c_void_p)
        MvCamera.sdk_dll.IMV_InternalWriteReg.restype = c_int
        # C原型:IMV_API int IMV_CALL IMV_InternalWriteReg(IN IMV_HANDLE handle, IN uint64_t regAddress, IN uint64_t regValue, IN_OUT unsigned int* pLength);
        return MvCamera.sdk_dll.IMV_InternalWriteReg(self.handle, regAddress, regValue, byref(pLength))

    # ch:保存图像
    def IMV_SaveImageToFile(self, pstSaveFileParam):
        MvCamera.sdk_dll.IMV_SaveImageToFile.argtype = (c_void_p, c_void_p)
        MvCamera.sdk_dll.IMV_SaveImageToFile.restype = c_int
        #IMV_API int IMV_CALL IMV_SaveImageToFile(IN IMV_HANDLE handle, IN IMV_SaveImageToFileParam * pstSaveFileParam);
        return MvCamera.sdk_dll.IMV_SaveImageToFile(self.handle, byref(pstSaveFileParam))
