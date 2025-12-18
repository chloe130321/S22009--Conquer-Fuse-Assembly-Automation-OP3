import sys
import os
import json
import shutil
import traceback
import zipfile
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QSlider, QFileDialog, QMessageBox, QGroupBox, QLineEdit,
                             QDialog, QDateEdit, QFormLayout, QDialogButtonBox, QProgressDialog)
from PySide6.QtCore import Qt, QDate

# ==========================================
# 🔧 1. 路徑設定 (依照你的 OP3 專案修改)
# ==========================================
MODEL_BASE_DIR = r"C:\3-1_3-3\model"
CONFIG_FILE = r"S22009--Conquer-Fuse-Assembly-Automation-OP3\config.json"  # 放在同層目錄即可

# 圖片根目錄 (參照 op3_save_images.py)
IMG_ROOT_OP3_1 = r"C:\G_D_2\S22009--Conquer-Fuse-Assembly-Automation-OP3\picture"
IMG_ROOT_OP3_3 = r"C:\3-1_3-3\OP3-3_pictures"

class DateRangeDialog(QDialog):
    """ 彈出式視窗：選擇日期範圍 (樣式保持不變) """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("選擇匯出日期範圍")
        self.resize(450, 250) 

        # (這裡的樣式表保持原樣，為了版面整潔省略重複的 CSS Code，功能完全相同)
        self.setStyleSheet("""
            QDialog { background-color: #2b2b2b; color: #ffffff; font-family: 'Microsoft JhengHei UI', sans-serif; }
            QDateEdit { background-color: #3c3f41; color: #e0e0e0; border: 2px solid #555; border-radius: 5px; padding: 5px 10px; font-size: 18px; min-height: 35px; }
            QDateEdit:hover { border: 2px solid #4db6ac; }
            QDateEdit::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 40px; border-left-width: 1px; border-left-color: #555; border-left-style: solid; background-color: #333; }
            QDateEdit::down-arrow { width: 16px; height: 16px; image: none; border: 2px solid #aaa; border-top: 0; border-right: 0; transform: rotate(-45deg); margin-top: -3px; }
            QCalendarWidget QWidget { alternate-background-color: #444; }
            QCalendarWidget QAbstractItemView { background-color: #2b2b2b; color: white; font-size: 16px; selection-background-color: #4db6ac; selection-color: black; }
            QCalendarWidget QWidget#qt_calendar_navigationbar { background-color: #2b2b2b; min-height: 40px; }
            QCalendarWidget QToolButton { color: white; font-weight: bold; icon-size: 24px; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(25)
        layout.setContentsMargins(40, 40, 40, 40)

        form = QFormLayout()
        form.setVerticalSpacing(20)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        today = QDate.currentDate()

        self.start_date = QDateEdit()
        self.start_date.setDate(today)
        self.start_date.setCalendarPopup(True) 
        self.start_date.setDisplayFormat("yyyy-MM-dd")

        self.end_date = QDateEdit()
        self.end_date.setDate(today)
        self.end_date.setCalendarPopup(True)
        self.end_date.setDisplayFormat("yyyy-MM-dd")

        lbl_start = QLabel(" 開始日期 :")
        lbl_start.setStyleSheet("font-size: 16px; font-weight: bold;")
        lbl_end = QLabel(" 結束日期 :")
        lbl_end.setStyleSheet("font-size: 16px; font-weight: bold;")

        form.addRow(lbl_start, self.start_date)
        form.addRow(lbl_end, self.end_date)
        
        layout.addLayout(form)
        layout.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("匯出")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        
        buttons.setStyleSheet("QPushButton { background-color: #0277bd; color: white; border-radius: 5px; padding: 8px 20px; font-size: 16px; font-weight: bold; min-width: 80px; } QPushButton:hover { background-color: #0288d1; }")
        
        layout.addWidget(buttons)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

    def get_dates(self):
        return self.start_date.date().toPython(), self.end_date.date().toPython()

class SettingsEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OP3 AOI 參數設定工具") # 修改標題
        self.resize(650, 550)

        self.setStyleSheet("""
            QWidget { background-color: #2b2b2b; color: #ffffff; font-family: 'Microsoft JhengHei UI'; font-size: 14px; }
            QGroupBox { border: 1px solid #555; border-radius: 8px; margin-top: 10px; font-weight: bold; color: #ddd; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
            QPushButton { background-color: #0277bd; color: white; border-radius: 4px; font-weight: bold; padding: 8px; }
            QPushButton:hover { background-color: #0288d1; }
            QLineEdit { background-color: #444; color: #ccc; border: 1px solid #555; border-radius: 4px; padding: 5px; }
        """)
        
        if not os.path.exists(MODEL_BASE_DIR):
            os.makedirs(MODEL_BASE_DIR, exist_ok=True)
        
        self.config = self.load_config()
        self.init_ui()

    def load_config(self):
        print("[Log] 正在讀取設定檔...")
        
        # 1. 預設值 (避免檔案不存在時程式崩潰)
        config = {
            "confidence_threshold": 0.80, 
            "model_filename_23": "", 
            "model_filename_25": ""
        }
        
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                    # ★ 核心邏輯：把硬碟裡的舊資料合併進來
                    # 這樣如果硬碟裡有模型設定，這裡就會讀進來，不會是空白的
                    config.update(saved_data)
                    print(f"[Success] 設定檔讀取成功: {config}")
            except Exception as e:
                print(f"❌ 設定檔讀取失敗 (將使用預設值): {e}")
                # 注意：如果讀取失敗，config 會保持預設值 (空白模型)，
                # 這時候如果你按儲存，確實會把空白存進去。
                # 但通常只要 config.json 沒壞，這步都會成功。
        else:
            print(f"[Warning] 找不到設定檔 {CONFIG_FILE}，將使用預設值。")
            
        return config

    def save_config(self):
        # ★ 安全加強版儲存邏輯 ★
        # 我們不直接把 self.config 覆蓋過去，而是先讀一次最新的檔案，再合併我們的修改
        # 這樣可以避免「不小心刪掉其他設定」或「覆蓋掉我們沒動到的欄位」
        
        try:
            final_data = {}
            # 1. 先嘗試讀取硬碟上現有的檔案
            if os.path.exists(CONFIG_FILE):
                try:
                    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                        final_data = json.load(f)
                except:
                    # 如果讀取失敗，就用空的字典，稍後會被 self.config 補上
                    pass
            
            # 2. 把目前介面上的設定 (self.config) 更新進去
            # 這時候 self.config 裡面已經包含了：
            #   (a) 剛啟動時讀到的舊模型 (如果你沒動)
            #   (b) 你剛剛拉動的新信心度
            final_data.update(self.config)

            # 3. 寫入檔案
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, indent=4, ensure_ascii=False)
            
            QMessageBox.information(self, "成功", "✅ 設定已儲存！\n請重新啟動 AOI 主程式以生效。")
            
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"儲存失敗: {e}")

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("🛠️ OP3-1 / OP3-3 系統參數設定")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #4db6ac;")
        layout.addWidget(title)
        
        path_info = QLabel(f"🔒 模型存放位置: {MODEL_BASE_DIR}")
        path_info.setStyleSheet("color: #777; font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(path_info)

        # --- 1. 信心度設定 ---
        group_conf = QGroupBox("信心度門檻 (Confidence)")
        group_layout = QVBoxLayout(group_conf)
        h_slider_layout = QHBoxLayout()
        
        # 讀取信心度，預設 0.8
        current_conf = self.config.get("confidence_threshold", 0.8)
        
        self.lbl_conf = QLabel(f"{int(current_conf*100)}%")
        self.lbl_conf.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffeb3b; min-width: 60px; qproperty-alignment: AlignCenter;")
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(50, 99)
        self.slider.setValue(int(current_conf*100))
        self.slider.valueChanged.connect(lambda v: (
            self.lbl_conf.setText(f"{v}%"), 
            self.config.update({"confidence_threshold": v/100.0})
        ))
        
        h_slider_layout.addWidget(self.slider) 
        h_slider_layout.addWidget(self.lbl_conf)
        group_layout.addLayout(h_slider_layout)

        lbl_tips = QLabel("💡 說明：若 AI 的把握度低於此設定值，系統將強制判定為 NG。")
        lbl_tips.setStyleSheet("color: #aaa; font-size: 12px; margin-top: 5px;")
        group_layout.addWidget(lbl_tips)
        layout.addWidget(group_conf)

        # --- 2. 模型選擇與圖片匯出 ---
        group_model = QGroupBox("模型檔案管理 & 圖片匯出")
        model_layout = QVBoxLayout(group_model)

        # 🔧 針對 OP3-1 建立欄位
        # key_in_json 是指 config['models'] 裡面的 key
        self.create_row(model_layout, "OP3-1 相機", "op3_1", img_root=IMG_ROOT_OP3_1)
        
        # 🔧 針對 OP3-3 建立欄位
        self.create_row(model_layout, "OP3-3 相機", "op3_3", img_root=IMG_ROOT_OP3_3)
        
        layout.addWidget(group_model)

        # --- 3. 儲存按鈕 ---
        layout.addStretch()
        btn_save = QPushButton("💾 儲存設定 (Save Config)")
        btn_save.setStyleSheet("background-color: #2e7d32; font-size: 16px; height: 40px;")
        btn_save.clicked.connect(self.save_config)
        layout.addWidget(btn_save)

        self.setLayout(layout)

    def create_row(self, layout, label_text, model_key, img_root):
        """
        建立一行介面：標籤 + 模型檔名 + 匯入按鈕 + 匯出圖片按鈕
        """
        lbl = QLabel(label_text)
        lbl.setStyleSheet("color: #4db6ac; margin-top: 5px;")
        layout.addWidget(lbl)
        
        h_layout = QHBoxLayout()
        
        # 顯示檔名
        line_edit = QLineEdit()
        current_model = self.config.get("models", {}).get(model_key, "")
        line_edit.setText(current_model)
        line_edit.setReadOnly(True)
        line_edit.setPlaceholderText("尚未設定模型...")
        
        # 匯入模型按鈕
        btn_import = QPushButton("📂 匯入模型")
        btn_import.clicked.connect(lambda: self.import_model(model_key, line_edit))
        
        # 匯出圖片按鈕
        btn_export = QPushButton("📤 匯出圖片")
        btn_export.setStyleSheet("background-color: #d84315;")
        # 將 img_root 和 標籤名稱 傳入
        btn_export.clicked.connect(lambda: self.export_images(img_root, label_text))
        
        h_layout.addWidget(line_edit)
        h_layout.addWidget(btn_import)
        h_layout.addWidget(btn_export)
        
        layout.addLayout(h_layout)

    def import_model(self, config_key, line_edit):
        # 修正後的檔案選擇視窗 (避免之前的 TypeError)
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "選擇新模型檔案",       
            "",                   
            "Model Files (*.pth)" 
        )
        
        # ★ 安全機制：
        # 如果你按取消 (file_path 為空)，程式直接結束，什麼都不改。
        # 你的舊設定 (self.config[config_key]) 會維持原樣。
        if not file_path: 
            return

        try:
            filename = os.path.basename(file_path)
            target_path = os.path.join(MODEL_BASE_DIR, filename)
            
            if os.path.abspath(file_path) != os.path.abspath(target_path):
                shutil.copy2(file_path, target_path)
                msg = f"已將檔案複製到系統目錄:\n{filename}"
            else:
                msg = f"已選擇系統目錄內的檔案:\n{filename}"

            self.config[config_key] = filename
            line_edit.setText(filename)
            QMessageBox.information(self, "匯入成功", msg)
            
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"檔案複製失敗: {e}")

    # ==================== ⭐️ 關鍵修改：簡化的資料夾掃描邏輯 ====================
    def scan_images_by_date(self, root_dir, start_date, end_date):
        matched_files = []
        print(f"[Log] 開始掃描目錄: {root_dir}")
        
        if not os.path.exists(root_dir):
            print("[Error] 目錄不存在")
            return matched_files

        try:
            for file_name in os.listdir(root_dir):
                file_path = os.path.join(root_dir, file_name)
                
                if not os.path.isfile(file_path):
                    continue
                
                if not file_name.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                    continue

                try:
                    date_part = file_name[:8] 
                    file_date = datetime.strptime(date_part, "%Y%m%d").date()
                    
                    if start_date <= file_date <= end_date:
                        matched_files.append(file_path)
                except ValueError:
                    continue
                    
        except Exception as e:
            print(f"[Error] 掃描過程出錯: {e}")
            traceback.print_exc()
        
        return matched_files

    def export_images(self, root_dir, cam_name):
        """ 處理匯出圖片 """
        
        # 1. 檢查目錄
        if not os.path.exists(root_dir):
            QMessageBox.warning(self, "路徑錯誤", f"找不到圖片路徑：\n{root_dir}\n請確認硬碟或資料夾是否正確。")
            return

        # 2. 選擇日期
        dlg = DateRangeDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return 

        start_date, end_date = dlg.get_dates()
        if start_date > end_date:
            QMessageBox.warning(self, "日期錯誤", "開始日期不能晚於結束日期！")
            return

        # 3. 搜尋檔案
        QApplication.setOverrideCursor(Qt.WaitCursor)
        files_to_zip = self.scan_images_by_date(root_dir, start_date, end_date)
        QApplication.restoreOverrideCursor()

        if not files_to_zip:
            QMessageBox.information(self, "查無資料", f"在 {start_date} 到 {end_date} 之間\n沒有找到 {cam_name} 的照片。")
            return

        # 4. 存檔
        zip_name = f"{cam_name.replace(' ','')}_{start_date}_{end_date}.zip"
        save_path, _ = QFileDialog.getSaveFileName(self, "儲存壓縮檔", zip_name, "Zip Files (*.zip)")
        
        if not save_path:
            return

        # 5. 壓縮
        progress = QProgressDialog(f"正在打包 {len(files_to_zip)} 張圖片...", "取消", 0, len(files_to_zip), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        try:
            with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for i, file_path in enumerate(files_to_zip):
                    if progress.wasCanceled():
                        break
                    
                    # 保持日期資料夾結構 (例如: 2023-12-17/xxx.png)
                    rel_path = os.path.relpath(file_path, root_dir)
                    zf.write(file_path, rel_path)
                    
                    progress.setValue(i + 1)

            if not progress.wasCanceled():
                QMessageBox.information(self, "完成", f"✅ 匯出成功！\n共打包 {len(files_to_zip)} 張圖片。")
            else:
                if os.path.exists(save_path):
                    os.remove(save_path)

        except Exception as e:
            QMessageBox.critical(self, "匯出失敗", f"打包過程發生錯誤:\n{e}")
        finally:
            progress.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SettingsEditor()
    window.show()
    sys.exit(app.exec())