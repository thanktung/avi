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
        self.setWindowTitle("AOI Control Panel")
        self.resize(1000, 600)
        self._build_ui()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QGridLayout(central)

        # --- Left panel ---
        left = QtWidgets.QFrame()
        left.setFrameShape(QtWidgets.QFrame.StyledPanel)
        vbox = QtWidgets.QVBoxLayout(left)

        self.btn_connect = QtWidgets.QPushButton("Connect OPC UA")
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect")
        self.btn_start = QtWidgets.QPushButton("Start AOI")
        self.btn_stop = QtWidgets.QPushButton("Stop AOI")

        for b in [self.btn_connect, self.btn_disconnect, self.btn_start, self.btn_stop]:
            vbox.addWidget(b)
        vbox.addStretch()

        # --- Image display ---
        self.image_view = QtWidgets.QLabel("No image")
        self.image_view.setAlignment(QtCore.Qt.AlignCenter)
        self.image_view.setStyleSheet("background-color:#222; color:#aaa; border:1px solid #555;")

        # --- Log box ---
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("System log...")

        # --- Status ---
        self.status_label = QtWidgets.QLabel("Status: Disconnected")
        self.status_label.setStyleSheet("font-weight:bold; color:red;")

        # --- Layout setup ---
        layout.addWidget(left, 0, 0, 2, 1)
        layout.addWidget(self.image_view, 0, 1)
        layout.addWidget(self.log_box, 1, 1)
        layout.addWidget(self.status_label, 2, 0, 1, 2)

        layout.setRowStretch(0, 5)
        layout.setRowStretch(1, 2)

        # --- Connect signals ---
        self.btn_connect.clicked.connect(self.handle_connect)
        self.btn_disconnect.clicked.connect(self.handle_disconnect)
        self.btn_start.clicked.connect(self.handle_start)
        self.btn_stop.clicked.connect(self.handle_stop)

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
