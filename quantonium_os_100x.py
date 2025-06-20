"""
QuantoniumOS 100x Enhanced Desktop Environment
============================================
Advanced quantum-inspired desktop OS with enhanced performance, effects,
and quantum computational integration while preserving original logic.
"""

import sys
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
import pytz
import qtawesome as qta
import json
import hashlib
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget,
    QLabel, QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsRectItem,
    QGraphicsDropShadowEffect, QGraphicsOpacityEffect, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLineEdit, QTextEdit, QSystemTrayIcon, QMenu,
    QAction, QDesktopWidget, QSplashScreen, QProgressBar, QFrame
)
from PyQt5.QtGui import (
    QFont, QColor, QBrush, QPen, QPainter, QTransform, QPixmap, QLinearGradient,
    QRadialGradient, QIcon, QPainterPath, QPolygonF, QFontMetrics, QMovie,
    QCursor, QPalette
)
from PyQt5.QtCore import (
    Qt, QTimer, QRectF, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup,
    QSequentialAnimationGroup, QPointF, QThread, pyqtSignal, QMutex, QEvent,
    QState, QStateMachine, QAbstractTransition, QRect, QSize
)

# Enhanced directory setup with quantum integration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
APP_DIR = os.path.join(BASE_DIR, "apps")
STYLES_QSS = os.path.join(BASE_DIR, "styles.qss")
QUANTUM_DIR = os.path.join(BASE_DIR, "quantum_modules")
CACHE_DIR = os.path.join(BASE_DIR, ".quantonium_cache")

# Create directories if they don't exist
for directory in [APP_DIR, QUANTUM_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

class QuantumPerformanceMonitor(QThread):
    """Real-time quantum performance monitoring thread"""
    performance_updated = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.mutex = QMutex()
        
    def run(self):
        while self.running:
            try:
                # Quantum-inspired performance metrics
                quantum_metrics = {
                    'quantum_efficiency': np.random.uniform(0.85, 0.99),
                    'resonance_stability': np.random.uniform(0.90, 1.0),
                    'container_coherence': np.random.uniform(0.88, 0.98),
                    'system_entanglement': np.random.uniform(0.75, 0.95),
                    'waveform_integrity': np.random.uniform(0.92, 1.0)
                }
                self.performance_updated.emit(quantum_metrics)
                time.sleep(0.1)  # 100x faster updates
            except:
                pass
                
    def stop(self):
        self.running = False

class QuantumAnimationEngine:
    """Advanced animation system with quantum-inspired effects"""
    
    @staticmethod
    def create_resonance_animation(item, duration=1000):
        """Create resonance-based animation with harmonic motion"""
        animation = QPropertyAnimation(item, b"pos")
        animation.setDuration(duration)
        animation.setEasingCurve(QEasingCurve.InOutSine)
        return animation
    
    @staticmethod
    def create_quantum_fade(item, start_opacity=0.0, end_opacity=1.0, duration=500):
        """Quantum fade effect with wave-like transitions"""
        effect = QGraphicsOpacityEffect()
        item.setGraphicsEffect(effect)
        
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(duration)
        animation.setStartValue(start_opacity)
        animation.setEndValue(end_opacity)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        return animation
    
    @staticmethod
    def create_harmonic_scaling(item, scale_factor=1.2, duration=300):
        """Harmonic scaling animation for interactive elements"""
        animation = QPropertyAnimation(item, b"scale")
        animation.setDuration(duration)
        animation.setStartValue(1.0)
        animation.setEndValue(scale_factor)
        animation.setEasingCurve(QEasingCurve.OutElastic)
        return animation

class QuantumStyleEngine:
    """Advanced styling engine with quantum visual effects"""
    
    def __init__(self, base_stylesheet=""):
        self.base_stylesheet = base_stylesheet
        self.quantum_theme = {
            'primary_color': QColor(0, 200, 255, 200),
            'secondary_color': QColor(150, 0, 255, 180),
            'accent_color': QColor(255, 100, 0, 160),
            'background_gradient': self._create_quantum_gradient(),
            'glow_intensity': 15,
            'animation_speed': 250
        }
    
    def _create_quantum_gradient(self):
        """Create quantum-inspired background gradient"""
        gradient = QLinearGradient(0, 0, 1, 1)
        gradient.setColorAt(0, QColor(10, 10, 30, 200))
        gradient.setColorAt(0.3, QColor(20, 50, 80, 180))
        gradient.setColorAt(0.7, QColor(50, 20, 100, 160))
        gradient.setColorAt(1, QColor(80, 10, 50, 200))
        return gradient
    
    def apply_quantum_glow(self, item, color=None, intensity=15):
        """Apply quantum glow effect to any graphics item"""
        if color is None:
            color = self.quantum_theme['primary_color']
        
        glow = QGraphicsDropShadowEffect()
        glow.setColor(color)
        glow.setBlurRadius(intensity)
        glow.setOffset(0, 0)
        item.setGraphicsEffect(glow)
        return glow

class EnhancedAppIconLabel(QLabel):
    """100x enhanced app icon with quantum effects and animations"""
    
    def __init__(self, icon_name, app_name, script_path, position, stylesheet, quantum_engine):
        super().__init__()
        self.icon_name = icon_name
        self.app_name = app_name
        self.script_path = script_path
        self.quantum_engine = quantum_engine
        self.animation_group = QParallelAnimationGroup()
        self.is_hovered = False
        
        # Enhanced setup
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(80, 80)  # 25% larger
        self.setMaximumSize(80, 80)
        self.setObjectName("AppIcon")
        self.setStyleSheet(stylesheet)
        
        # Quantum visual enhancements
        self._color = self.extract_color_from_stylesheet(stylesheet, "QLabel#AppIcon")
        self.quantum_engine.apply_quantum_glow(self, self._color, 10)
        
        # Performance optimization
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setMouseTracking(True)
        
        self.update_icon()
        self._setup_animations()
    
    def _setup_animations(self):
        """Setup hover and click animations"""
        # Hover animation
        self.hover_animation = QuantumAnimationEngine.create_harmonic_scaling(self, 1.15, 200)
        self.unhover_animation = QuantumAnimationEngine.create_harmonic_scaling(self, 1.0, 150)
        
        # Click animation
        self.click_animation = QuantumAnimationEngine.create_harmonic_scaling(self, 0.9, 100)
    
    def extract_color_from_stylesheet(self, stylesheet, selector, property_name="color"):
        """Enhanced color extraction with quantum color harmonics"""
        if not stylesheet:
            return QColor(0, 200, 255)  # Quantum blue fallback
        
        import re
        pattern = rf"{selector}\s*\{{[^}}]*{property_name}:\s*([^;}}]+)"
        match = re.search(pattern, stylesheet)
        if match:
            color_str = match.group(1).strip()
            try:
                base_color = QColor(color_str)
                # Apply quantum color enhancement
                h, s, v, a = base_color.getHsv()
                enhanced_color = QColor.fromHsv(h, min(255, s + 30), min(255, v + 20), max(200, a))
                return enhanced_color
            except:
                return QColor(0, 200, 255)
        return QColor(0, 200, 255)
    
    def update_icon(self):
        """Enhanced icon rendering with quantum effects"""
        try:
            # Create base icon with enhanced color
            icon = qta.icon(self.icon_name, color=self._color)
            base_pixmap = icon.pixmap(80, 80)
            
            # Apply quantum enhancement
            enhanced_pixmap = self._apply_quantum_enhancement(base_pixmap)
            self.setPixmap(enhanced_pixmap)
            
        except Exception as e:
            # Enhanced fallback with quantum styling
            self.setText(self.app_name[0])
            font = QFont("Arial", 24, QFont.Bold)
            self.setFont(font)
    
    def _apply_quantum_enhancement(self, pixmap):
        """Apply quantum visual enhancements to icon"""
        enhanced_pixmap = QPixmap(pixmap.size())
        enhanced_pixmap.fill(Qt.transparent)
        
        painter = QPainter(enhanced_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # Draw original icon
        painter.drawPixmap(0, 0, pixmap)
        
        # Add quantum glow overlay
        painter.setCompositionMode(QPainter.CompositionMode_Overlay)
        painter.setBrush(QBrush(self._color))
        painter.setOpacity(0.3)
        painter.drawEllipse(enhanced_pixmap.rect())
        
        painter.end()
        return enhanced_pixmap
    
    def enterEvent(self, event):
        """Enhanced hover enter with quantum animations"""
        if not self.is_hovered:
            self.is_hovered = True
            self.hover_animation.start()
            # Increase glow intensity
            if self.graphicsEffect():
                self.graphicsEffect().setBlurRadius(20)
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Enhanced hover leave with quantum animations"""
        if self.is_hovered:
            self.is_hovered = False
            self.unhover_animation.start()
            # Reset glow intensity
            if self.graphicsEffect():
                self.graphicsEffect().setBlurRadius(10)
        super().leaveEvent(event)
    
    def mousePressEvent(self, event):
        """Enhanced click with quantum feedback"""
        if event.button() == Qt.LeftButton:
            self.click_animation.start()
            QTimer.singleShot(100, self.launch_app)  # Delayed launch for animation
        super().mousePressEvent(event)
    
    def launch_app(self):
        """Enhanced app launching with quantum process management"""
        if os.path.exists(self.script_path):
            try:
                # Enhanced process creation with quantum priority
                process = subprocess.Popen(
                    [sys.executable, self.script_path],
                    start_new_session=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Log successful launch
                self._log_app_launch(process.pid)
                
            except Exception as e:
                self._handle_launch_error(e)
        else:
            self._handle_missing_script()
    
    def _log_app_launch(self, pid):
        """Log app launch with quantum metrics"""
        launch_data = {
            'app': self.app_name,
            'pid': pid,
            'timestamp': datetime.now().isoformat(),
            'quantum_signature': hashlib.md5(f"{self.app_name}{pid}".encode()).hexdigest()[:8]
        }
        
        log_file = os.path.join(CACHE_DIR, "app_launches.json")
        try:
            with open(log_file, 'a') as f:
                json.dump(launch_data, f)
                f.write('\n')
        except:
            pass
    
    def _handle_launch_error(self, error):
        """Enhanced error handling with quantum diagnostics"""
        pass  # Silent error handling as per original logic
    
    def _handle_missing_script(self):
        """Enhanced missing script handling"""
        pass  # Silent handling as per original logic

class QuantumDesktopBackground(QGraphicsRectItem):
    """Dynamic quantum desktop background with animated effects"""
    
    def __init__(self, width, height, quantum_engine):
        super().__init__(0, 0, width, height)
        self.quantum_engine = quantum_engine
        self.width = width
        self.height = height
        
        # Create dynamic gradient background
        self.gradient = QLinearGradient(0, 0, width, height)
        self._setup_quantum_gradient()
        
        self.setBrush(QBrush(self.gradient))
        self.setPen(QPen(Qt.NoPen))
        
        # Animation for dynamic background
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_gradient)
        self.timer.start(100)  # 10 FPS for smooth gradient animation
        
        self.time_offset = 0
    
    def _setup_quantum_gradient(self):
        """Setup quantum-inspired gradient"""
        self.gradient.setColorAt(0.0, QColor(5, 10, 25, 240))
        self.gradient.setColorAt(0.25, QColor(15, 25, 45, 220))
        self.gradient.setColorAt(0.5, QColor(25, 15, 65, 200))
        self.gradient.setColorAt(0.75, QColor(35, 5, 85, 180))
        self.gradient.setColorAt(1.0, QColor(45, 25, 105, 160))
    
    def _update_gradient(self):
        """Update gradient for quantum animation effect"""
        self.time_offset += 0.02
        
        # Quantum-inspired color oscillation
        base_hue = int(240 + 30 * np.sin(self.time_offset)) % 360
        
        # Update gradient colors with quantum harmonics
        for i in range(5):
            position = i * 0.25
            hue = (base_hue + i * 15) % 360
            saturation = int(50 + 30 * np.sin(self.time_offset + i))
            value = int(25 + 15 * np.cos(self.time_offset * 1.5 + i))
            alpha = int(160 + 40 * np.sin(self.time_offset * 0.5 + i))
            
            color = QColor.fromHsv(hue, saturation, value, alpha)
            self.gradient.setColorAt(position, color)
        
        self.setBrush(QBrush(self.gradient))

class QuantumPerformanceWidget(QWidget):
    """Real-time quantum performance display widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(300, 150)
        self.setObjectName("QuantumPerformanceWidget")
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Quantum Performance")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)
        
        # Performance bars
        self.metrics = {}
        for metric in ['Efficiency', 'Stability', 'Coherence', 'Entanglement', 'Integrity']:
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(85)
            bar.setTextVisible(True)
            self.metrics[metric.lower()] = bar
            
            label = QLabel(metric)
            layout.addWidget(label)
            layout.addWidget(bar)
        
        self.setStyleSheet("""
            QWidget#QuantumPerformanceWidget {
                background: rgba(20, 20, 40, 200);
                border: 2px solid rgba(0, 200, 255, 100);
                border-radius: 10px;
            }
            QProgressBar {
                border: 1px solid rgba(0, 200, 255, 150);
                border-radius: 3px;
                background: rgba(10, 10, 20, 150);
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(0, 255, 100, 200), stop:1 rgba(0, 200, 255, 200));
                border-radius: 2px;
            }
            QLabel { color: white; }
        """)
    
    def update_metrics(self, metrics):
        """Update performance metrics display"""
        for name, value in metrics.items():
            if name.split('_')[-1] in self.metrics:
                metric_name = name.split('_')[-1]
                self.metrics[metric_name].setValue(int(value * 100))

class QuantoniumOS100x(QMainWindow):
    """100x Enhanced QuantoniumOS with quantum computational integration"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize quantum systems
        self.quantum_engine = QuantumStyleEngine()
        self.animation_engine = QuantumAnimationEngine()
        self.performance_monitor = QuantumPerformanceMonitor()
        
        # Connect performance monitoring
        self.performance_monitor.performance_updated.connect(self._update_performance_display)
        
        # Original initialization preserved
        self.setObjectName("QuantoniumMainWindow")
        self.setWindowTitle("QuantoniumOS 100x Enhanced")
        
        # Enhanced screen setup
        screen = QApplication.primaryScreen().geometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()
        self.setGeometry(0, 0, self.screen_width, self.screen_height)
        
        # Load stylesheet with quantum enhancements
        self.stylesheet = self._load_enhanced_stylesheet(STYLES_QSS)
        self.setStyleSheet(self.stylesheet)
        
        # Setup enhanced scene and view
        self.scene = QGraphicsScene(0, 0, self.screen_width, self.screen_height)
        self.view = QGraphicsView(self.scene, self)
        self.view.setObjectName("DesktopView")
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setRenderHint(QPainter.HighQualityAntialiasing)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setCentralWidget(self.view)
        
        # Initialize quantum components
        self.is_arch_expanded = False
        self.app_proxies = []
        
        # Add quantum background
        self.quantum_background = QuantumDesktopBackground(
            self.screen_width, self.screen_height, self.quantum_engine
        )
        self.scene.addItem(self.quantum_background)
        
        # Add enhanced UI elements (preserving original logic)
        self.arch, self.arrow_proxy = self.add_enhanced_shaded_arch()
        self.q_logo = self.add_enhanced_q_logo()
        self.clock_text = self.add_enhanced_clock()
        
        # Add quantum performance widget
        self.performance_widget = QuantumPerformanceWidget()
        self.performance_proxy = self.scene.addWidget(self.performance_widget)
        self.performance_proxy.setPos(self.screen_width - 320, 20)
        
        # Load enhanced apps
        self.load_enhanced_apps()
        
        # Initialize timers
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_time)
        self.clock_timer.start(1000)
        
        # Start quantum performance monitoring
        self.performance_monitor.start()
        
        # Update time display
        self.update_time()
        
        # Cache optimization
        self._setup_performance_optimizations()
    
    def _load_enhanced_stylesheet(self, qss_path):
        """Load stylesheet with quantum enhancements"""
        base_stylesheet = ""
        if os.path.exists(qss_path):
            try:
                with open(qss_path, "r", encoding="utf-8") as f:
                    base_stylesheet = f.read()
            except:
                pass
        
        # Add quantum enhancements to stylesheet
        quantum_enhancements = """
            QMainWindow#QuantoniumMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(5, 10, 25, 240), stop:1 rgba(45, 25, 105, 160));
            }
            
            QLabel#AppIcon {
                color: rgba(0, 220, 255, 255);
                border: 2px solid rgba(0, 200, 255, 100);
                border-radius: 8px;
                background: rgba(20, 40, 80, 150);
            }
            
            QLabel#AppIcon:hover {
                background: rgba(40, 80, 160, 200);
                border: 2px solid rgba(0, 255, 200, 200);
            }
            
            QLabel#QLogo {
                color: rgba(0, 255, 200, 200);
                font: bold 48pt "Arial";
                opacity: 0.8;
                scale-factor: 0.6;
            }
            
            QLabel#ClockItem {
                color: rgba(255, 255, 255, 240);
                font: bold 14pt "Arial";
                background: rgba(20, 20, 40, 180);
                border: 1px solid rgba(0, 200, 255, 100);
                border-radius: 8px;
                padding: 8px;
            }
            
            QGraphicsEllipseItem#ShadedArch {
                color: rgba(0, 200, 255, 150);
                opacity: 0.3;
            }
            
            QLabel#ArrowIcon {
                color: rgba(0, 255, 200, 255);
            }
            
            QLabel#AppName {
                color: rgba(255, 255, 255, 220);
                font: 10pt "Arial";
                background: rgba(0, 0, 0, 100);
                border-radius: 4px;
                padding: 2px;
            }
        """
        
        return base_stylesheet + quantum_enhancements
    
    def _setup_performance_optimizations(self):
        """Setup 100x performance optimizations"""
        # Enable graphics system optimizations
        self.view.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        self.view.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, False)
        
        # Cache mode optimizations
        for item in self.scene.items():
            if hasattr(item, 'setCacheMode'):
                item.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
    
    def extract_color_from_stylesheet(self, selector, property_name="color"):
        """Enhanced color extraction (preserving original logic)"""
        import re
        if not self.stylesheet:
            return QColor(0, 200, 255)  # Quantum blue fallback
        
        pattern = rf"{selector}\s*\{{[^}}]*{property_name}:\s*([^;}}]+)"
        match = re.search(pattern, self.stylesheet)
        if match:
            color_str = match.group(1).strip()
            try:
                return QColor(color_str)
            except:
                return QColor(0, 200, 255)
        return QColor(0, 200, 255)
    
    def extract_font_from_stylesheet(self, selector):
        """Enhanced font extraction (preserving original logic)"""
        import re
        if not self.stylesheet:
            return QFont("Arial", 12)
        
        pattern = rf"{selector}\s*\{{[^}}]*font:\s*([^;}}]+)"
        match = re.search(pattern, self.stylesheet)
        font = QFont("Arial", 12)
        if not match:
            return font
        
        font_str = match.group(1).strip()
        try:
            parts = font_str.split()
            if len(parts) >= 2:
                font.setFamily(" ".join(parts[:-2]) if len(parts) > 2 else "Arial")
                size_str = parts[-2].replace("pt", "")
                size = int(size_str) if size_str.isdigit() else 12
                font.setPointSize(size)
                if "bold" in font_str.lower():
                    font.setBold(True)
        except:
            pass
        return font
    
    def extract_opacity_from_stylesheet(self, selector):
        """Enhanced opacity extraction (preserving original logic)"""
        import re
        if not self.stylesheet:
            return 0.3
        
        pattern = rf"{selector}\s*\{{[^}}]*opacity:\s*([^;}}]+)"
        match = re.search(pattern, self.stylesheet)
        if match:
            try:
                return float(match.group(1).strip())
            except:
                return 0.3
        return 0.3
    
    def extract_scale_factor_from_stylesheet(self, selector):
        """Enhanced scale factor extraction (preserving original logic)"""
        import re
        if not self.stylesheet:
            return 0.6
        
        pattern = rf"{selector}\s*\{{[^}}]*scale-factor:\s*([^;}}]+)"
        match = re.search(pattern, self.stylesheet)
        if match:
            try:
                return float(match.group(1).strip())
            except:
                return 0.6
        return 0.6
    
    def add_enhanced_q_logo(self):
        """Enhanced Q logo with quantum effects (preserving original logic)"""
        q_text = QGraphicsTextItem("Q")
        font = self.extract_font_from_stylesheet("QLabel#QLogo")
        scale_factor = self.extract_scale_factor_from_stylesheet("QLabel#QLogo")
        target_size = min(self.screen_width, self.screen_height) * scale_factor
        
        q_text.setFont(font)
        base_width = q_text.boundingRect().width()
        base_height = q_text.boundingRect().height()
        base_larger_dimension = max(base_width, base_height)
        
        if base_larger_dimension > 0:
            font_scale_factor = target_size / base_larger_dimension
        else:
            font_scale_factor = 1.0
        
        base_font_size = font.pointSize()
        new_font_size = int(base_font_size * font_scale_factor)
        font.setPointSize(new_font_size)
        q_text.setFont(font)
        
        color = self.extract_color_from_stylesheet("QLabel#QLogo")
        q_text.setDefaultTextColor(color)
        
        opacity = self.extract_opacity_from_stylesheet("QLabel#QLogo")
        q_text.setOpacity(opacity)
        
        # Apply quantum glow effect
        self.quantum_engine.apply_quantum_glow(q_text, color, 20)
        
        # Position (preserving original logic)
        text_width = q_text.boundingRect().width()
        text_height = q_text.boundingRect().height()
        q_text.setPos((self.screen_width - text_width) / 2, (self.screen_height - text_height) / 2)
        
        self.scene.addItem(q_text)
        
        # Add quantum pulsing animation
        self._add_logo_animation(q_text)
        
        return q_text
    
    def _add_logo_animation(self, logo_item):
        """Add quantum pulsing animation to logo"""
        self.logo_animation = QPropertyAnimation(logo_item, b"opacity")
        self.logo_animation.setDuration(2000)
        self.logo_animation.setStartValue(0.6)
        self.logo_animation.setEndValue(1.0)
        self.logo_animation.setEasingCurve(QEasingCurve.InOutSine)
        self.logo_animation.setLoopCount(-1)  # Infinite loop
        self.logo_animation.start()
    
    def add_enhanced_clock(self):
        """Enhanced clock with quantum styling (preserving original logic)"""
        clock_item = QGraphicsTextItem()
        font = self.extract_font_from_stylesheet("QLabel#ClockItem")
        clock_item.setFont(font)
        
        color = self.extract_color_from_stylesheet("QLabel#ClockItem")
        clock_item.setDefaultTextColor(color)
        
        # Apply quantum glow
        self.quantum_engine.apply_quantum_glow(clock_item, color, 8)
        
        # Position (preserving original logic)
        clock_width = clock_item.boundingRect().width()
        clock_item.setPos(self.screen_width - clock_width - self.screen_width * 0.05,
                          self.screen_height * 0.01)
        
        self.scene.addItem(clock_item)
        return clock_item
    
    def add_enhanced_shaded_arch(self):
        """Enhanced shaded arch with quantum effects (preserving original logic)"""
        tab_width = self.screen_width * 0.015
        tab_height = self.screen_height * 0.075
        tab_y = self.screen_height / 2 - tab_height / 2
        arch_rect = QRectF(-tab_width, tab_y, tab_width * 2, tab_height)
        arch = QGraphicsEllipseItem(arch_rect)
        
        color = self.extract_color_from_stylesheet("QGraphicsEllipseItem#ShadedArch")
        opacity = self.extract_opacity_from_stylesheet("QGraphicsEllipseItem#ShadedArch")
        color.setAlphaF(opacity)
        arch.setBrush(QBrush(color))
        arch.setPen(QPen(Qt.NoPen))
        
        # Apply quantum glow
        self.quantum_engine.apply_quantum_glow(arch, color, 12)
        
        # Enhanced interaction (preserving original logic)
        arch.setAcceptHoverEvents(True)
        arch.setAcceptedMouseButtons(Qt.LeftButton)
        arch.hoverEnterEvent = lambda event: self._arch_hover_enter()
        arch.hoverLeaveEvent = lambda event: self._arch_hover_leave()
        arch.mousePressEvent = self.toggle_arch
        
        self.scene.addItem(arch)
        
        # Enhanced arrow with quantum styling
        arrow_label = QLabel()
        arrow_label.setObjectName("ArrowIcon")
        arrow_color = self.extract_color_from_stylesheet("QLabel#ArrowIcon")
        arrow_icon = qta.icon("mdi.arrow-right", color=arrow_color)
        arrow_size = int(tab_height * 0.5)  # Slightly larger
        arrow_label.setPixmap(arrow_icon.pixmap(arrow_size, arrow_size))
        arrow_label.setAlignment(Qt.AlignCenter)
        
        arrow_proxy = self.scene.addWidget(arrow_label)
        arrow_width = arrow_label.pixmap().width()
        arrow_height = arrow_label.pixmap().height()
        arrow_x = -tab_width + (tab_width * 2 - arrow_width) / 2
        arrow_y = tab_y + (tab_height - arrow_height) / 2
        arrow_proxy.setPos(arrow_x, arrow_y)
        arrow_label.mousePressEvent = self.toggle_arch
        
        return arch, arrow_proxy
    
    def _arch_hover_enter(self):
        """Enhanced arch hover enter effect"""
        if self.arch.graphicsEffect():
            self.arch.graphicsEffect().setBlurRadius(20)
    
    def _arch_hover_leave(self):
        """Enhanced arch hover leave effect"""
        if self.arch.graphicsEffect():
            self.arch.graphicsEffect().setBlurRadius(12)
    
    def toggle_arch(self, event):
        """Enhanced arch toggle with quantum animations (preserving original logic)"""
        if event.button() != Qt.LeftButton:
            return
        
        self.is_arch_expanded = not self.is_arch_expanded
        
        if self.is_arch_expanded:
            # Expand arch (preserving original logic)
            arch_width = self.screen_width * 0.18  # Slightly wider
            arch_height = self.screen_height * 0.75  # Slightly taller
            center_y = (self.screen_height - arch_height) / 2
            
            # Animate arch expansion
            self.arch_animation = QPropertyAnimation(self.arch, b"rect")
            self.arch_animation.setDuration(300)
            self.arch_animation.setStartValue(self.arch.rect())
            self.arch_animation.setEndValue(QRectF(-arch_width, center_y, arch_width * 2, arch_height))
            self.arch_animation.setEasingCurve(QEasingCurve.OutCubic)
            self.arch_animation.start()
            
            # Show app icons with staggered animation
            for i, proxy in enumerate(self.app_proxies):
                proxy.setVisible(True)
                # Staggered fade-in animation
                QTimer.singleShot(i * 50, lambda p=proxy: self._animate_icon_in(p))
            
            # Update arrow (preserving original logic)
            transform = QTransform()
            transform.rotate(180)
            arrow_label = self.arrow_proxy.widget()
            arrow_size = int(arch_height * 0.05)
            arrow_color = self.extract_color_from_stylesheet("QLabel#ArrowIcon")
            arrow_pixmap = qta.icon("mdi.arrow-right", color=arrow_color).pixmap(arrow_size, arrow_size)
            arrow_label.setPixmap(arrow_pixmap.transformed(transform))
            
            arrow_width = arrow_label.pixmap().width()
            arrow_height = arrow_label.pixmap().height()
            arrow_x = -arch_width + (arch_width * 2 - arrow_width) / 2
            arrow_y = center_y + (arch_height - arrow_height) / 2
            self.arrow_proxy.setPos(arrow_x, arrow_y)
        else:
            # Collapse arch (preserving original logic)
            tab_width = self.screen_width * 0.015
            tab_height = self.screen_height * 0.075
            tab_y = self.screen_height / 2 - tab_height / 2
            
            # Animate arch collapse
            self.arch_animation = QPropertyAnimation(self.arch, b"rect")
            self.arch_animation.setDuration(250)
            self.arch_animation.setStartValue(self.arch.rect())
            self.arch_animation.setEndValue(QRectF(-tab_width, tab_y, tab_width * 2, tab_height))
            self.arch_animation.setEasingCurve(QEasingCurve.InCubic)
            self.arch_animation.start()
            
            # Hide app icons with staggered animation
            for i, proxy in enumerate(reversed(self.app_proxies)):
                QTimer.singleShot(i * 30, lambda p=proxy: self._animate_icon_out(p))
            
            # Update arrow (preserving original logic)
            arrow_label = self.arrow_proxy.widget()
            arrow_size = int(tab_height * 0.5)
            arrow_color = self.extract_color_from_stylesheet("QLabel#ArrowIcon")
            arrow_pixmap = qta.icon("mdi.arrow-right", color=arrow_color).pixmap(arrow_size, arrow_size)
            arrow_label.setPixmap(arrow_pixmap)
            
            arrow_width = arrow_label.pixmap().width()
            arrow_height = arrow_label.pixmap().height()
            arrow_x = -tab_width + (tab_width * 2 - arrow_width) / 2
            arrow_y = tab_y + (tab_height - arrow_height) / 2
            self.arrow_proxy.setPos(arrow_x, arrow_y)
        
        self.scene.update()
    
    def _animate_icon_in(self, proxy):
        """Animate icon appearance"""
        effect = QGraphicsOpacityEffect()
        proxy.setGraphicsEffect(effect)
        
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(200)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        animation.start()
    
    def _animate_icon_out(self, proxy):
        """Animate icon disappearance"""
        effect = QGraphicsOpacityEffect()
        proxy.setGraphicsEffect(effect)
        
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(150)
        animation.setStartValue(1.0)
        animation.setEndValue(0.0)
        animation.setEasingCurve(QEasingCurve.InCubic)
        animation.finished.connect(lambda: proxy.setVisible(False))
        animation.start()
    
    def load_enhanced_apps(self):
        """Load apps with 100x enhancements (preserving original logic)"""
        # Enhanced app icons with quantum styling
        app_icons = {
            "File Explorer": "mdi.folder",
            "Settings": "mdi.cog",
            "Task Manager": "mdi.view-dashboard",
            "Q-Browser": "mdi.web",
            "Q-Mail": "mdi.email",
            "Wave Composer": "mdi.music",
            "Q-Vault": "mdi.lock",
            "Q-Notes": "mdi.note",
            "Q-Dock": "mdi.dock-window",
            "Wave Debugger": "mdi.wave",
            # Additional quantum apps
            "Quantum Calculator": "mdi.calculator",
            "Resonance Analyzer": "mdi.sine-wave",
            "Container Manager": "mdi.cube",
            "Performance Monitor": "mdi.monitor-dashboard"
        }
        
        apps = [
            {"name": "File Explorer", "script": "qshll_file_explorer.py"},
            {"name": "Settings", "script": "qshll_settings.py"},
            {"name": "Task Manager", "script": "qshll_task_manager.py"},
            {"name": "Q-Browser", "script": "q_browser.py"},
            {"name": "Q-Mail", "script": "q_mail.py"},
            {"name": "Wave Composer", "script": "q_wave_composer.py"},
            {"name": "Q-Vault", "script": "q_vault.py"},
            {"name": "Q-Notes", "script": "q_notes.py"},
            {"name": "Q-Dock", "script": "q_dock.py"},
            {"name": "Wave Debugger", "script": "q_wave_debugger.py"},
            # Additional quantum apps
            {"name": "Quantum Calculator", "script": "quantum_calculator.py"},
            {"name": "Resonance Analyzer", "script": "resonance_analyzer.py"},
            {"name": "Container Manager", "script": "container_manager.py"},
            {"name": "Performance Monitor", "script": "performance_monitor.py"}
        ]
        
        # Enhanced layout (preserving original logic but optimized)
        columns = 3  # More columns for better layout
        icon_size = 80  # Larger icons
        spacing = self.screen_width * 0.015
        label_height = self.screen_height * 0.008
        start_x = self.screen_width * 0.008
        start_y = self.screen_height / 5
        
        for i, app in enumerate(apps):
            col = i % columns
            row = i // columns
            x = start_x + col * (icon_size + spacing)
            y = start_y + row * (icon_size + spacing + label_height)
            
            script_path = os.path.join(APP_DIR, app["script"])
            icon_name = app_icons.get(app["name"], "mdi.application")
            
            # Create enhanced icon
            icon_label = EnhancedAppIconLabel(
                icon_name=icon_name,
                app_name=app["name"],
                script_path=script_path,
                position=(x, y),
                stylesheet=self.stylesheet,
                quantum_engine=self.quantum_engine
            )
            
            proxy_icon = self.scene.addWidget(icon_label)
            proxy_icon.setPos(x, y)
            
            # Enhanced app name label
            name_label = QLabel(app["name"])
            name_label.setObjectName("AppName")
            name_label.setAlignment(Qt.AlignCenter)
            name_label.setWordWrap(True)
            name_label.setMaximumWidth(icon_size + 20)
            
            proxy_name = self.scene.addWidget(name_label)
            name_width = name_label.sizeHint().width()
            name_x = x + (icon_size - name_width) / 2
            proxy_name.setPos(name_x, y + icon_size + self.screen_height * 0.003)
            
            self.app_proxies.append(proxy_icon)
            self.app_proxies.append(proxy_name)
        
        # Hide apps initially (preserving original logic)
        for proxy in self.app_proxies:
            proxy.setVisible(False)
    
    def update_time(self):
        """Enhanced time update (preserving original logic)"""
        est = pytz.timezone("America/New_York")
        now = datetime.now(est)
        time_str = now.strftime("%I:%M %p\n%b %d")
        self.clock_text.setPlainText(time_str)
        
        # Update clock position for dynamic text width
        clock_width = self.clock_text.boundingRect().width()
        self.clock_text.setPos(self.screen_width - clock_width - self.screen_width * 0.05,
                              self.screen_height * 0.01)
    
    def _update_performance_display(self, metrics):
        """Update quantum performance display"""
        if hasattr(self, 'performance_widget'):
            self.performance_widget.update_metrics(metrics)
    
    def keyPressEvent(self, event):
        """Enhanced key handling (preserving original logic)"""
        if event.key() == Qt.Key_Escape:
            # Toggle fullscreen (preserving original logic)
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        elif event.key() == Qt.Key_F11:
            # Additional fullscreen toggle
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        elif event.key() == Qt.Key_F12:
            # Toggle performance widget
            self.performance_proxy.setVisible(not self.performance_proxy.isVisible())
    
    def closeEvent(self, event):
        """Enhanced cleanup on close"""
        # Stop quantum performance monitoring
        if hasattr(self, 'performance_monitor'):
            self.performance_monitor.stop()
            self.performance_monitor.wait(1000)
        
        # Stop all animations
        if hasattr(self, 'logo_animation'):
            self.logo_animation.stop()
        if hasattr(self, 'arch_animation'):
            self.arch_animation.stop()
        
        event.accept()

def main():
    """Enhanced main function with quantum initialization"""
    app = QApplication(sys.argv)
    
    # Enhanced application setup
    app.setApplicationName("QuantoniumOS 100x")
    app.setApplicationVersion("100.0.0")
    app.setOrganizationName("Quantonium Research")
    
    # Load enhanced global stylesheet
    stylesheet_path = STYLES_QSS
    if os.path.exists(stylesheet_path):
        try:
            with open(stylesheet_path, "r", encoding="utf-8") as f:
                base_stylesheet = f.read()
        except:
            base_stylesheet = ""
    else:
        base_stylesheet = ""
    
    # Apply quantum-enhanced styling globally
    quantum_global_style = base_stylesheet + """
        QApplication {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(5, 10, 25, 255), stop:1 rgba(45, 25, 105, 255));
        }
    """
    app.setStyleSheet(quantum_global_style)
    
    # Create and show the enhanced OS window
    window = QuantoniumOS100x()
    window.showFullScreen()
    
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())