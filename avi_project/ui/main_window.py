from PyQt5 import QtWidgets, QtCore, QtGui
import cv2, os

# === Giả lập các hàm xử lý OPC và AOI ===
def connect_opcua():
    print("🔌 OPC UA connected")
    return True

def disconnect_opcua():
    print("❌ OPC UA disconnected")
    return True

def send_start_signal():
    print("▶️ Start signal sent to PLC")

def send_stop_signal():
    print("⏹ Stop signal sent to PLC")

def get_processed_image():
    # trả về path ảnh kết quả hoặc ảnh trong RAM
    return "test_images/sample.png"  # ví dụ


# === GUI chính ===
class AOI_GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AOI Control Panel") #Tên
        self.resize(1000, 600) #Kích thước

        self._build_menu()
        self._build_ui() #Gọi hàm build giao diện

    def _build_menu(self):

        ### Build menu
        menu = self.menuBar()

        # ----- File menu -----
        file_menu = menu.addMenu("File")

        act_open = QtWidgets.QAction("Open Image", self)
        act_save = QtWidgets.QAction("Save Image", self)
        act_exit = QtWidgets.QAction("Exit", self)

        file_menu.addAction(act_open)
        file_menu.addAction(act_save)
        file_menu.addSeparator()
        file_menu.addAction(act_exit)

        # ----- Device menu -----
        device_menu = menu.addMenu("Device")

        act_cam = QtWidgets.QAction("Connect Camera", self)
        act_plc = QtWidgets.QAction("Connect PLC", self)

        device_menu.addAction(act_cam)
        device_menu.addAction(act_plc)

        # ----- Help menu -----
        help_menu = menu.addMenu("Help")

        act_about = QtWidgets.QAction("About", self)
        help_menu.addAction(act_about)

        # Connect signals
        act_exit.triggered.connect(self.close)
        act_open.triggered.connect(self.menu_open_file)


    def _build_ui(self):
        # Main screen
        central = QtWidgets.QWidget() # tạo widget rỗng
        self.setCentralWidget(central) # đưa widget làm cửa sổ chính
        layout = QtWidgets.QGridLayout(central) # tạo bố cục layout

        # --- Left panel ---
        left = QtWidgets.QFrame() # tạo 1 khung
        left.setFrameShape(QtWidgets.QFrame.StyledPanel) # khung có viền
        vbox = QtWidgets.QVBoxLayout(left) # tạo layout dọc cho bên trái, gán layout vào left

        self.btn_connect = QtWidgets.QPushButton("Connect OPC UA") # tạo 4 nút
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect")
        self.btn_start = QtWidgets.QPushButton("Start AOI")
        self.btn_stop = QtWidgets.QPushButton("Stop AOI")

        for b in [self.btn_connect, self.btn_disconnect, self.btn_start, self.btn_stop]: # thêm 4 nút vào layout
            vbox.addWidget(b)
        vbox.addStretch()# thêm khoảng trống đàn hồi phía dưới đẩy nút lên trên

        # --- Right panel (PLC status) ---
        right = QtWidgets.QFrame()
        right.setFrameShape(QtWidgets.QFrame.StyledPanel)
        rvbox = QtWidgets.QVBoxLayout(right)

        self.lbl_plc_conn = QtWidgets.QLabel("PLC: Disconnected")
        self.lbl_plc_mode = QtWidgets.QLabel("Mode: -")
        self.lbl_plc_trigger = QtWidgets.QLabel("Trigger: -")
        self.lbl_plc_result = QtWidgets.QLabel("Last Result: -")

        for w in [self.lbl_plc_conn, self.lbl_plc_mode, self.lbl_plc_trigger, self.lbl_plc_result]:
            rvbox.addWidget(w)

        rvbox.addStretch()

        # --- Image display ---
        self.image_view = QtWidgets.QLabel("No image") #dùng Qlable để hiển thị ảnh
        self.image_view.setAlignment(QtCore.Qt.AlignCenter) # căn chữ/ảnh vào chính giữa
        self.image_view.setStyleSheet("background-color:#222; color:#aaa; border:1px solid #555;") #style giao diện

        # --- Log box ---
        self.log_box = QtWidgets.QPlainTextEdit() #widget hiển thị nhiều dòng text
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("System log...")

        # --- Status ---
        self.status_label = QtWidgets.QLabel("Status: Disconnected")
        self.status_label.setStyleSheet("font-weight:bold; color:red;")

        # --- Layout setup ---
        layout.addWidget(left, 0, 0, 2, 1) # đặt widget tạo hàng 0 cột 0 chiếm 2 hàng 1 cột
        layout.addWidget(self.image_view, 0, 1) # đặt widget ở hàng 0 cột 1, chiếm 1 hàng 1 cột mặc định
        layout.addWidget(self.log_box, 1, 1) #đặt ở hàng 1 cột 1 chiếm 1 hàng 1 cột
        layout.addWidget(self.status_label, 2, 0, 1, 2) # đặt ở hàng 2 cột 0 chiếm 1 hàng 2 cột
        layout.addWidget(right, 0, 2, 2, 1)

        layout.setRowStretch(0, 5)# tỉ trọng khi stretch
        layout.setRowStretch(1, 2)

        # --- Connect signals ---
        self.btn_connect.clicked.connect(self.handle_connect)
        self.btn_disconnect.clicked.connect(self.handle_disconnect)
        self.btn_start.clicked.connect(self.handle_start)
        self.btn_stop.clicked.connect(self.handle_stop)

    def menu_open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*.*)"
        )

        if not path:
            return  # user bấm Cancel

        # --- lưu path nếu cần ---
        self.current_image_path = path

        # --- log ra ---
        self.log_box.appendPlainText(f"Selected image: {path}")

        # --- load & hiển thị ảnh ---
        pix = QtGui.QPixmap(path)

        if pix.isNull():
            self.log_box.appendPlainText("Error: Cannot load image.")
            return

        # scale ảnh vừa khung nhưng giữ tỷ lệ
        pix = pix.scaled(
            self.image_view.width(),
            self.image_view.height(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )

        self.image_view.setPixmap(pix)

    # ==== LOGIC HANDLERS ====
    def handle_connect(self):
        if connect_opcua():
            self.status_label.setText("Status: Connected ✅")
            self.status_label.setStyleSheet("color:green; font-weight:bold;")
            self.log("Connected to OPC UA server.")

    def handle_disconnect(self):
        if disconnect_opcua():
            self.status_label.setText("Status: Disconnected ❌")
            self.status_label.setStyleSheet("color:red; font-weight:bold;")
            self.log("Disconnected from OPC UA server.")

    def handle_start(self):
        send_start_signal()
        self.log("AOI started. Running image processing...")
        self.show_image(get_processed_image())

    def handle_stop(self):
        send_stop_signal()
        self.log("AOI stopped.")

    def show_image(self, img_path):
        if not os.path.exists(img_path):
            self.log(f"Image not found: {img_path}")
            return
        pix = QtGui.QPixmap(img_path)
        pix = pix.scaled(self.image_view.width(), self.image_view.height(), QtCore.Qt.KeepAspectRatio)
        self.image_view.setPixmap(pix)
        self.log(f"Displayed image: {img_path}")

    def log(self, text):
        self.log_box.appendPlainText(text)
        print(text)


# === MAIN ===
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    gui = AOI_GUI()
    gui.show()
    app.exec_()