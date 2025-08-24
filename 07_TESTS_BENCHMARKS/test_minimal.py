import sys

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow


class TestApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuantoniumOS Test App")
        self.setGeometry(100, 100, 400, 300)
        label = QLabel("Hello from QuantoniumOS!", self)
        label.setGeometry(50, 50, 300, 200)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestApp()
    window.show()
    sys.exit(app.exec_())
