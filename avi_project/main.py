import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow
from core.algorithm import Pipeline

if __name__ == "__main__":
    app = QApplication(sys.argv)

    pipeline_avi = Pipeline()               # tạo logic
    win = MainWindow(pipeline_avi)       # truyền logic vào UI
    win.show()

    sys.exit(app.exec_())
