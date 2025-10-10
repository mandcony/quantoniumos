#!/usr/bin/env python3
"""
QuantoniumOS Simple Intro Animation
=================================
Clean, centered animation that fits perfectly in the window
"""

import sys
import math
import random
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsView, QGraphicsScene, 
                            QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsLineItem,
                            QGraphicsPathItem)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QBrush, QRadialGradient, QPainterPath

class SimpleQuantoniumIntro(QMainWindow):
    """Simple, working intro animation with spiral background"""
    
    animation_finished = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        # Constants
        self.phi = 1.618033988749895
        
        # Setup window
        self.setWindowTitle("QuantoniumOS")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.showFullScreen()
        self.setStyleSheet("background-color: #0b0b0b;")
        
        # Get screen size
        screen = QApplication.desktop().screenGeometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()
        self.center_x = self.screen_width / 2
        self.center_y = self.screen_height / 2
        
        # Setup graphics - CRITICAL: Set scene size FIRST
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, self.screen_width, self.screen_height)
        
        self.view = QGraphicsView(self.scene, self)
        
        # REMOVE SCROLL BARS - This is the key fix
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setFrameStyle(0)
        
        # Set rendering
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        
        self.setCentralWidget(self.view)
        
        # Make sure view matches scene exactly
        self.view.setSceneRect(0, 0, self.screen_width, self.screen_height)
        
        # Animation components
        self.spiral_lines = []
        self.logo_elements = []
        self.text_elements = []
        
        # Setup animation
        self.setup_spiral()
        self.setup_logo()
        self.setup_text()
        
        # Animation state
        self.animation_step = 0
        self.current_phase = "spiral"
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
    
    def setup_spiral(self):
        """Create spiral lines in center"""
        spiral_count = 8
        max_radius = 200  # Keep contained
        
        for i in range(spiral_count):
            # Create spiral path
            path = QPainterPath()
            
            # Start from center and spiral outward
            points = 50
            for j in range(points):
                t = j / points
                # Golden spiral formula
                angle = i * (math.pi * 2 / spiral_count) + t * math.pi * 4
                radius = t * max_radius
                
                x = self.center_x + radius * math.cos(angle)
                y = self.center_y + radius * math.sin(angle)
                
                if j == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            
            # Create graphics item
            spiral_line = QGraphicsPathItem(path)
            
            # Style the spiral
            pen = QPen(QColor("#2471a3"), 2)
            pen.setCapStyle(Qt.RoundCap)
            spiral_line.setPen(pen)
            spiral_line.setOpacity(0)
            
            self.scene.addItem(spiral_line)
            self.spiral_lines.append(spiral_line)
    
    def setup_logo(self):
        """Create simple Q logo in center"""
        # Logo size - keep reasonable
        outer_radius = 60
        inner_radius = 40
        stroke_width = 3
        
        # Outer circle
        self.outer_circle = QGraphicsEllipseItem(
            self.center_x - outer_radius,
            self.center_y - outer_radius,
            outer_radius * 2,
            outer_radius * 2
        )
        
        # Inner circle
        self.inner_circle = QGraphicsEllipseItem(
            self.center_x - inner_radius,
            self.center_y - inner_radius,
            inner_radius * 2,
            inner_radius * 2
        )
        
        # Q dash
        dash_length = outer_radius * 0.6
        dash_start_x = self.center_x + inner_radius * 0.3
        dash_start_y = self.center_y + inner_radius * 0.3
        dash_end_x = dash_start_x + dash_length * 0.7
        dash_end_y = dash_start_y + dash_length * 0.7
        
        self.q_dash = QGraphicsLineItem(dash_start_x, dash_start_y, dash_end_x, dash_end_y)
        
        # Style all elements
        pen = QPen(QColor("#3498db"), stroke_width)
        pen.setCapStyle(Qt.RoundCap)
        
        for element in [self.outer_circle, self.inner_circle]:
            element.setPen(pen)
            element.setBrush(QBrush(QColor(0, 0, 0, 0)))
            element.setOpacity(0)
            self.scene.addItem(element)
        
        self.q_dash.setPen(pen)
        self.q_dash.setOpacity(0)
        self.scene.addItem(self.q_dash)
        
        self.logo_elements = [self.outer_circle, self.inner_circle, self.q_dash]
    
    def setup_text(self):
        """Create text elements"""
        # Title
        self.title = QGraphicsTextItem("QUANTONIUMOS")
        title_font = QFont("Arial", 24, QFont.Bold)
        self.title.setFont(title_font)
        self.title.setDefaultTextColor(QColor("#3498db"))
        
        # Center title below logo
        title_bounds = self.title.boundingRect()
        title_x = self.center_x - (title_bounds.width() / 2)
        title_y = self.center_y + 100
        self.title.setPos(title_x, title_y)
        self.title.setOpacity(0)
        
        # Subtitle
        self.subtitle = QGraphicsTextItem("Symbolic Quantum-Inspired Computing")
        subtitle_font = QFont("Arial", 14, QFont.Normal)
        self.subtitle.setFont(subtitle_font)
        self.subtitle.setDefaultTextColor(QColor("#85c1e9"))
        
        # Center subtitle below title
        subtitle_bounds = self.subtitle.boundingRect()
        subtitle_x = self.center_x - (subtitle_bounds.width() / 2)
        subtitle_y = title_y + title_bounds.height() + 10
        self.subtitle.setPos(subtitle_x, subtitle_y)
        self.subtitle.setOpacity(0)
        
        self.scene.addItem(self.title)
        self.scene.addItem(self.subtitle)
        
        self.text_elements = [self.title, self.subtitle]
    
    def start_animation(self):
        """Start the animation"""
        print("🌀 Starting spiral quantum intro animation...")
        self.timer.start(50)  # 20 FPS
    
    def update_animation(self):
        """Update animation frame by frame"""
        self.animation_step += 1
        elapsed = self.animation_step * 50
        
        if self.current_phase == "spiral":
            self.update_spiral(elapsed)
        elif self.current_phase == "logo":
            self.update_logo(elapsed)
        elif self.current_phase == "text":
            self.update_text(elapsed)
        elif self.current_phase == "complete":
            self.timer.stop()
            self.animation_finished.emit()
    
    def update_spiral(self, elapsed):
        """Animate the spiral background - fade in then fade out"""
        fade_in_duration = 1500
        hold_duration = 1000
        fade_out_duration = 1000
        total_duration = fade_in_duration + hold_duration + fade_out_duration
        
        for i, spiral in enumerate(self.spiral_lines):
            delay = i * 80
            if elapsed > delay:
                adjusted_elapsed = elapsed - delay
                
                if adjusted_elapsed < fade_in_duration:
                    # Fade in phase
                    progress = adjusted_elapsed / fade_in_duration
                    pulse = 0.3 + 0.4 * math.sin(elapsed * 0.003 + i * 0.5)
                    spiral.setOpacity(progress * pulse)
                
                elif adjusted_elapsed < fade_in_duration + hold_duration:
                    # Hold phase - keep visible with pulse
                    pulse = 0.5 + 0.3 * math.sin(elapsed * 0.003 + i * 0.5)
                    spiral.setOpacity(pulse)
                
                elif adjusted_elapsed < total_duration:
                    # Fade out phase
                    fade_progress = (adjusted_elapsed - fade_in_duration - hold_duration) / fade_out_duration
                    opacity = (1 - fade_progress) * 0.8
                    pulse = 0.3 + 0.2 * math.sin(elapsed * 0.003 + i * 0.5)
                    spiral.setOpacity(opacity * pulse)
                
                else:
                    # Completely faded out
                    spiral.setOpacity(0)
        
        # Move to logo phase after spiral fades out
        if elapsed > total_duration + 500:
            self.current_phase = "logo"
            self.animation_step = 0
    
    def update_logo(self, elapsed):
        """Show logo"""
        duration = 1000
        
        progress = min(1.0, elapsed / duration)
        for element in self.logo_elements:
            element.setOpacity(progress)
        
        if elapsed > duration:
            self.current_phase = "text"
            self.animation_step = 0
    
    def update_text(self, elapsed):
        """Show text"""
        # Title
        if elapsed > 0:
            title_progress = min(1.0, elapsed / 800)
            self.title.setOpacity(title_progress)
        
        # Subtitle
        if elapsed > 300:
            subtitle_progress = min(1.0, (elapsed - 300) / 800)
            self.subtitle.setOpacity(subtitle_progress)
        
        if elapsed > 2000:
            self.current_phase = "complete"
    
    def keyPressEvent(self, event):
        """ESC to skip"""
        if event.key() == Qt.Key_Escape:
            self.timer.stop()
            self.animation_finished.emit()
        super().keyPressEvent(event)

def main():
    """Test the intro"""
    app = QApplication(sys.argv)
    
    intro = SimpleQuantoniumIntro()
    intro.show()
    intro.start_animation()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
