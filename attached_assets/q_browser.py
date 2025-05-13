import sys
import os
import logging
import re
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QLineEdit,
    QPushButton, QHBoxLayout, QStatusBar, QTabWidget, QMenu, QFrame, QTabBar
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, Qt, QRect, QPoint, QPropertyAnimation, QEasingCurve

# Add the parent directory (C:\quantonium_os) to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import Quantonium OS Resonance Functions
try:
    from quantum_entanglement import encrypt_resonance, decrypt_resonance, get_resonance_weight
except ImportError as e:
    logging.error(f"Failed to import quantum_entanglement: {e}")
    raise

try:
    from symbolic_eigenvector import encode_resonance, decode_resonance, compute_similarity
except ImportError as e:
    logging.error(f"Failed to import symbolic_eigenvector: {e}")
    raise

# Define logging settings
APP_DIR = r"C:\quantonium_os\apps"
log_dir = APP_DIR if os.access(APP_DIR, os.W_OK) else r"C:\temp"
log_file = os.path.join(log_dir, "q_browser.log")
logging.basicConfig(level=logging.DEBUG, filename=log_file, filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

# Path to external QSS file
STYLES_QSS = r"C:\quantonium_os\styles.qss"

def load_stylesheet(qss_path):
    """Load and return the contents of the QSS file if it exists."""
    if os.path.exists(qss_path):
        with open(qss_path, "r") as f:
            return f.read()
    else:
        logger.warning(f"QSS file not found: {qss_path}")
        return ""

class CustomTabBar(QTabBar):
    """Tab bar with integrated new tab button using symbolic resonance states."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.newTabBtn = QPushButton("+")
        self.newTabBtn.setFixedSize(20, 20)
        self.newTabBtn.clicked.connect(parent.addNewTabBtn if parent else lambda: None)
        self.newTabBtn.setStyleSheet("""
            QPushButton {
                border: 1px solid #ccc;
                border-radius: 5px;
                background-color: #f0f0f0;
                font-size: 12px;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
        """)

    def tabSizeHint(self, index):
        size = super().tabSizeHint(index)
        return size

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updateNewTabButtonPosition()

    def updateNewTabButtonPosition(self):
        if self.count() > 0:
            last_tab_rect = self.tabRect(self.count() - 1)
            x = last_tab_rect.right() + 2
            y = last_tab_rect.y() + (last_tab_rect.height() - self.newTabBtn.height()) // 2
            self.newTabBtn.move(x, y)
        else:
            self.newTabBtn.move(2, 2)

class QBrowser(QMainWindow):
    def __init__(self):
        super().__init__()
        try:
            self.setWindowTitle("Q-Browser")
            self.setGeometry(100, 100, 1024, 768)

            # Main widget with a layout
            self.centralWidget = QWidget(self)
            self.setCentralWidget(self.centralWidget)
            self.mainLayout = QVBoxLayout(self.centralWidget)
            self.mainLayout.setContentsMargins(0, 0, 0, 0)

            # Navigation bar at the top
            self.navFrame = QFrame(self)
            self.navFrame.setStyleSheet("""
                QFrame {
                    background-color: #f0f0f0;
                    border-bottom: 1px solid #ccc;
                }
            """)
            self.navFrame.setFixedHeight(40)
            nav_layout = QHBoxLayout(self.navFrame)
            nav_layout.setContentsMargins(10, 5, 10, 5)

            self.backBtn = QPushButton("Back")
            self.backBtn.setObjectName("backBtn")
            self.backBtn.clicked.connect(self.goBack)
            nav_layout.addWidget(self.backBtn)

            self.forwardBtn = QPushButton("Forward")
            self.forwardBtn.setObjectName("forwardBtn")
            self.forwardBtn.clicked.connect(self.goForward)
            nav_layout.addWidget(self.forwardBtn)

            self.urlEntry = QLineEdit()
            self.urlEntry.setText("https://duckduckgo.com")
            self.urlEntry.returnPressed.connect(self.navigateToUrl)
            nav_layout.addWidget(self.urlEntry)

            self.goBtn = QPushButton("Go")
            self.goBtn.clicked.connect(self.navigateToUrl)
            nav_layout.addWidget(self.goBtn)

            self.bookmarkBtn = QPushButton("Bookmark")
            self.bookmarkBtn.clicked.connect(self.addBookmark)
            nav_layout.addWidget(self.bookmarkBtn)

            self.historyBtn = QPushButton("History")
            self.historyBtn.clicked.connect(self.showHistory)
            nav_layout.addWidget(self.historyBtn)

            self.mainLayout.addWidget(self.navFrame)

            # Tab widget with Resonance-based Sorting
            self.tabWidget = QTabWidget()
            self.tabWidget.setTabsClosable(True)
            self.tabWidget.tabCloseRequested.connect(self.closeTab)
            self.tabWidget.setTabBar(CustomTabBar(self))
            self.tabWidget.currentChanged.connect(self.animateTabTransition)
            self.mainLayout.addWidget(self.tabWidget)

            # Add first tab
            self.addNewTab("https://duckduckgo.com", "New Tab")

            self.statusBar = QStatusBar()
            self.setStatusBar(self.statusBar)
            logger.info("Q-Browser UI fully initialized")

            # Bookmarks list
            self.bookmarks = []

            # Update navigation buttons initially
            self.updateNavigationButtons()

        except Exception as e:
            logger.error(f"Error initializing Q-Browser: {e}")
            raise

    def addNewTab(self, url, title):
        browser = QWebEngineView()
        # Use the original URL for navigation, not the encoded one
        browser.setUrl(QUrl(url))
        browser.urlChanged.connect(lambda qurl: self.updateUrlBar(qurl, browser))
        browser.urlChanged.connect(self.updateNavigationButtons)
        index = self.tabWidget.addTab(browser, title)
        self.tabWidget.setCurrentIndex(index)
        self.tabWidget.tabBar().updateNewTabButtonPosition()
        self.log_navigation(url)

    def addNewTabBtn(self):
        """Add a new tab when the '+' button is clicked."""
        self.addNewTab("https://duckduckgo.com", "New Tab")
        logger.info("Added new tab")

    def closeTab(self, index):
        if self.tabWidget.count() > 1:
            self.tabWidget.removeTab(index)
        self.tabWidget.tabBar().updateNewTabButtonPosition()

    def updateUrlBar(self, qurl, browser=None):
        if browser == self.tabWidget.currentWidget():
            # Decode the URL for display, but keep original for navigation
            decoded_url = decode_resonance(qurl.toString())
            self.urlEntry.setText(decoded_url)

    def updateNavigationButtons(self):
        current_browser = self.tabWidget.currentWidget()
        if current_browser:
            self.backBtn.setEnabled(current_browser.history().canGoBack())
            self.forwardBtn.setEnabled(current_browser.history().canGoForward())

    def navigateToUrl(self):
        try:
            url = self.urlEntry.text().strip()
            if not url:
                return
            # Basic URL validation
            if not re.match(r"^(https?://)([a-zA-Z0-9.-]+)(:[0-9]+)?(/.*)?$", url):
                logger.warning(f"Blocked unsafe URL: {url}")
                return
            if not url.startswith(("http://", "https://")):
                url = "https://duckduckgo.com/?q=" + url
            # Use the original URL for navigation
            current_browser = self.tabWidget.currentWidget()
            if current_browser:
                current_browser.setUrl(QUrl(url))
                logger.info(f"Navigated to URL: {url}")
                self.log_navigation(url)
        except Exception as e:
            logger.error(f"Error navigating to URL: {e}")

    def goBack(self):
        current_browser = self.tabWidget.currentWidget()
        if current_browser and current_browser.history().canGoBack():
            current_browser.back()

    def goForward(self):
        current_browser = self.tabWidget.currentWidget()
        if current_browser and current_browser.history().canGoForward():
            current_browser.forward()

    def addBookmark(self):
        current_browser = self.tabWidget.currentWidget()
        url = decode_resonance(current_browser.url().toString()) if current_browser else self.urlEntry.text()
        if url:
            self.bookmarks.append(encode_resonance(url))  # Encode for storage
            logger.info(f"Bookmarked URL (encoded): {encode_resonance(url)}")
            self.sortBookmarks()

    def sortBookmarks(self):
        current_url = self.tabWidget.currentWidget().url().toString() if self.tabWidget.currentWidget() else "https://duckduckgo.com"
        decoded_current_url = decode_resonance(current_url)
        # Sort bookmarks by resonance similarity
        try:
            self.bookmarks.sort(key=lambda url: compute_similarity(decode_resonance(url), decoded_current_url), reverse=True)
            logger.info("Sorted bookmarks based on resonance similarity.")
        except Exception as e:
            logger.error(f"Error sorting bookmarks: {e}")

    def log_navigation(self, url):
        try:
            encrypted_url = encrypt_resonance(url)
            with open(log_file, "a") as f:
                f.write(f"{encrypted_url}\n")
            logger.info(f"Securely logged navigation to: {encrypted_url}")
        except Exception as e:
            logger.error(f"Error logging navigation: {e}")

    def showHistory(self):
        try:
            with open(log_file, "r") as f:
                encrypted_history = f.readlines()
            decrypted_history = [decode_resonance(line.strip()) for line in encrypted_history if line.strip()]
            history_menu = QMenu("History", self)
            for url in decrypted_history:
                history_menu.addAction(url, lambda u=url: self.navigateToUrl(u))
            history_menu.exec_(self.historyBtn.mapToGlobal(self.historyBtn.rect().bottomLeft()))
            logger.info(f"Displayed history: {decrypted_history}")
        except Exception as e:
            logger.error(f"Error showing history: {e}")

    def animateTabTransition(self, index):
        tab = self.tabWidget.widget(index)
        if tab:
            anim = QPropertyAnimation(tab, b"pos")
            anim.setDuration(500)
            anim.setEasingCurve(QEasingCurve.InOutQuad)
            anim.setStartValue(QPoint(tab.pos().x(), tab.pos().y()))
            anim.setEndValue(QPoint(tab.pos().x(), tab.pos().y() + 10))
            anim.start()

if __name__ == "__main__":
    try:
        logger.info("Starting QApplication")
        app = QApplication(sys.argv)

        # Load and apply the external QSS stylesheet
        stylesheet = load_stylesheet(STYLES_QSS)
        if stylesheet:
            app.setStyleSheet(stylesheet)
        else:
            default_stylesheet = """
                QLineEdit {
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    padding: 5px;
                    background-color: #ffffff;
                    font-size: 14px;
                    margin: 0 5px;
                    min-width: 200px;
                }
                QPushButton {
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    background-color: #ffffff;
                    font-size: 12px;
                    padding: 5px 10px;
                    margin: 0 5px;
                }
                QPushButton#backBtn, QPushButton#forwardBtn {
                    background-color: #e0e0e0;
                }
                QPushButton:hover {
                    background-color: #d0d0d0;
                }
                QTabWidget::pane {
                    border: none;
                    background: transparent;
                }
                QTabBar::tab {
                    background: #f0f0f0;
                    padding: 5px 10px;
                    margin-right: 2px;
                    border-top-left-radius: 5px;
                    border-top-right-radius: 5px;
                }
                QTabBar::tab:selected {
                    background: #ffffff;
                }
            """
            app.setStyleSheet(default_stylesheet)

        browser = QBrowser()
        browser.show()
        logger.info("Q-Browser window shown")
        sys.exit(app.exec())
    except Exception as e:
        logger.error(f"Error in Q-Browser main: {e}")
        sys.exit(1)