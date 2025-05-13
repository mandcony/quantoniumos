import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileSystemModel, QTreeView, QHBoxLayout, QWidget,
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsProxyWidget,
    QVBoxLayout, QSizePolicy, QLabel
)
from PyQt5.QtGui import QBrush, QColor, QPen, QPainter, QLinearGradient, QFont
from PyQt5.QtCore import Qt, QTimer, QRectF, QDir

# Path to external QSS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STYLES_QSS = os.path.join(SCRIPT_DIR, "styles.qss")

def load_stylesheet(qss_path):
    if os.path.exists(qss_path):
        with open(qss_path, "r") as f:
            return f.read()
    else:
        print(f"⚠️ QSS file not found: {qss_path}")
        return ""

class QSHLLFileExplorer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QSHLL File Explorer")
        self.setGeometry(100, 100, 900, 600)

        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.mainLayout = QHBoxLayout(self.centralWidget)
        self.mainLayout.setContentsMargins(20, 20, 20, 20)
        self.mainLayout.setSpacing(15)
        self.mainLayout.setStretch(0, 1)
        self.mainLayout.setStretch(1, 2)

        # Arch Container with Root Folders
        self.scene = QGraphicsScene(0, 0, 300, 600)
        self.view = QGraphicsView(self.scene)
        self.view.setObjectName("ArchView")
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setFixedWidth(300)

        arch_width = 150
        arch_height = 600
        arch_rect = QRectF(-arch_width, 0, arch_width * 2, arch_height)
        self.arch = QGraphicsEllipseItem(arch_rect)
        gradient = QLinearGradient(-arch_width, 0, arch_width, arch_height)
        gradient.setColorAt(0, QColor(138, 154, 158, 100))
        gradient.setColorAt(1, QColor(138, 154, 158, 50))
        self.arch.setBrush(QBrush(gradient))
        self.arch.setPen(QPen(Qt.NoPen))
        self.scene.addItem(self.arch)

        # Add "PROCESS / PERFORMANCE" label at the top
        process_label = QLabel("PROCESS / PERFORMANCE")
        process_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        process_label.setStyleSheet("color: #2E3A4A; background: transparent;")
        process_proxy = self.scene.addWidget(process_label)
        process_proxy.setPos(-arch_width + 10, 10)

        # Add "FILES" and "ROOT FOLDERS" labels
        files_label = QLabel("FILES")
        files_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        files_label.setStyleSheet("color: #2E3A4A; background: transparent;")
        files_proxy = self.scene.addWidget(files_label)
        files_proxy.setPos(-arch_width + 10, 50)

        root_folders_label = QLabel("ROOT FOLDERS")
        root_folders_label.setFont(QFont("Segoe UI", 10))
        root_folders_label.setStyleSheet("color: #6D7A84; background: transparent;")
        root_folders_proxy = self.scene.addWidget(root_folders_label)
        root_folders_proxy.setPos(-arch_width + 10, 80)

        self.rootModel = QFileSystemModel()
        root_path = os.path.expanduser("~")
        if not os.path.exists(root_path):
            print(f"Error: Root path {root_path} does not exist! Falling back to C:\\")
            root_path = "C:\\"
        else:
            print(f"✅ Root path set to: {root_path}")
        self.rootModel.setRootPath(root_path)
        try:
            self.rootModel.setFilter(QDir.Dirs | QDir.NoDotAndDotDot)
        except AttributeError as e:
            print(f"Error setting filter: {e}")
            self.rootModel.setFilter(QDir.AllEntries)

        self.rootTreeView = QTreeView()
        self.rootTreeView.setModel(self.rootModel)
        root_index = self.rootModel.index(root_path)
        if not root_index.isValid():
            print(f"Error: Invalid root index for path {root_path}, trying parent")
            root_path = os.path.dirname(root_path)
            root_index = self.rootModel.index(root_path)
        self.rootTreeView.setRootIndex(root_index)
        self.rootTreeView.setHeaderHidden(True)
        self.rootTreeView.setAnimated(True)
        self.rootTreeView.clicked.connect(self.update_file_list)
        self.rootTreeView.setFixedWidth(200)
        self.rootTreeView.setFixedHeight(400)
        self.rootTreeView.setStyleSheet("""
            QTreeView {
                background-color: rgba(255, 255, 255, 0.8);
                border: none;
                border-radius: 10px;
                padding: 5px;
            }
            QTreeView::item:selected {
                background-color: #A3C6C4;
                color: #2E3A4A;
            }
        """)

        proxy_widget = self.scene.addWidget(self.rootTreeView)
        try:
            proxy_widget.setPos(-arch_width + 20, 110)
            proxy_widget.setMaximumWidth(200)
            proxy_widget.setMaximumHeight(arch_height - 200)
            if proxy_widget.sceneBoundingRect().right() > arch_width or proxy_widget.sceneBoundingRect().left() < -arch_width:
                print("Warning: Proxy widget exceeds arch bounds, adjusting position")
                proxy_widget.setPos(-arch_width + 10, 110)
        except Exception as e:
            print(f"Error positioning proxy widget: {e}")

        self.mainLayout.addWidget(self.view, 1)

        # File List Section (VIEW FILELIST)
        self.fileWidget = QWidget()
        self.fileWidget.setStyleSheet("""
            QWidget {
                background-color: #F5F1E9;
                border-radius: 10px;
            }
        """)
        self.fileLayout = QVBoxLayout(self.fileWidget)
        self.fileLayout.setContentsMargins(15, 15, 15, 15)
        self.fileLayout.setSpacing(10)

        file_label = QLabel("VIEW FILELIST")
        file_label.setAlignment(Qt.AlignCenter)
        file_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        file_label.setStyleSheet("color: #2E3A4A; background: transparent;")
        self.fileLayout.addWidget(file_label)

        self.fileModel = QFileSystemModel()
        self.fileModel.setRootPath(root_path)

        self.fileTreeView = QTreeView()
        self.fileTreeView.setModel(self.fileModel)
        file_index = self.fileModel.index(root_path)
        if not file_index.isValid():
            print(f"Error: Invalid file index for path {root_path}, trying parent")
            root_path = os.path.dirname(root_path)
            file_index = self.fileModel.index(root_path)
        self.fileTreeView.setRootIndex(file_index)
        self.fileTreeView.setHeaderHidden(True)
        self.fileTreeView.setAnimated(True)
        self.fileTreeView.setSortingEnabled(True)
        self.fileTreeView.sortByColumn(0, Qt.AscendingOrder)
        self.fileTreeView.setStyleSheet("""
            QTreeView {
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                padding: 5px;
            }
            QTreeView::item:hover {
                background-color: #E0E0E0;
            }
        """)
        self.fileLayout.addWidget(self.fileTreeView)
        self.fileWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.mainLayout.addWidget(self.fileWidget, 2)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_lists)
        self.timer.start(5000)

    def update_file_list(self, index):
        try:
            path = self.rootModel.filePath(index)
            if not os.path.exists(path):
                print(f"Error: Path {path} does not exist!")
                return
            print(f"Attempting to update file list to path: {path}")
            self.fileModel.setRootPath(path)
            file_index = self.fileModel.index(path)
            if not file_index.isValid():
                print(f"Warning: Invalid index for path {path}, using parent")
                path = os.path.dirname(path)
                file_index = self.fileModel.index(path)
            self.fileTreeView.setRootIndex(file_index)
            self.fileTreeView.sortByColumn(0, Qt.AscendingOrder)
            print(f"File list updated to path: {path}")
        except Exception as e:
            print(f"❌ Error updating file list: {e}")

    def refresh_lists(self):
        if self.isVisible():
            try:
                current_root_path = self.rootModel.rootPath()
                if not os.path.exists(current_root_path):
                    print(f"Error: Root path {current_root_path} does not exist!")
                    return
                self.rootModel.setRootPath("")
                self.rootModel.setRootPath(current_root_path)
                root_index = self.rootModel.index(current_root_path)
                if not root_index.isValid():
                    print(f"Error: Invalid root index for path {current_root_path}")
                    root_index = self.rootModel.index(os.path.dirname(current_root_path))
                self.rootTreeView.setRootIndex(root_index)

                current_file_path = self.fileModel.rootPath()
                if not os.path.exists(current_file_path):
                    print(f"Error: File path {current_file_path} does not exist!")
                    return
                self.fileModel.setRootPath("")
                self.fileModel.setRootPath(current_file_path)
                file_index = self.fileModel.index(current_file_path)
                if not file_index.isValid():
                    print(f"Error: Invalid file index for path {current_file_path}")
                    file_index = self.fileModel.index(os.path.dirname(current_file_path))
                self.fileTreeView.setRootIndex(file_index)
                self.fileTreeView.sortByColumn(0, Qt.AscendingOrder)
                print("✅ Root folders and file list refreshed (real-time update)")
            except Exception as e:
                print(f"❌ Error refreshing lists: {e}")

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    qss_path = STYLES_QSS
    stylesheet = load_stylesheet(qss_path)
    if stylesheet:
        print("✅ Stylesheet loaded successfully.")
    else:
        print("⚠️ Stylesheet could not be loaded.")
    app.setStyleSheet(stylesheet)
    fileExplorer = QSHLLFileExplorer()
    fileExplorer.show()
    sys.exit(app.exec_())