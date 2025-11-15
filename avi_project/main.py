import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainUI
from core.pipeline import main_pipeline

if __name__ == "__main__":
    app = QApplication(sys.argv)

    pipeline_avi = main_pipeline()               # tạo logic
    win = MainUI(pipeline_avi)       # truyền logic vào UI
    win.show()

    sys.exit(app.exec_())
