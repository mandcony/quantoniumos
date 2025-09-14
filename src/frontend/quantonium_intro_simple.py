#!/usr/bin/env python3
"""
QuantoniumOS Intro Animation - Simplified Working Version
========================================================
Advanced quantum field animation with proper rendering
"""

import sys
import math
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsView, QGraphicsScene, 
                            QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsLineItem)
from PyQt5.QtCore import (Qt, QTimer, pyqtSignal, QPointF)
from PyQt5.QtGui import (QColor, QFont, QPainter, QPen, QBrush, 
                        QRadialGradient, QGuiApplication)

class QuantumParticle:
    """Quantum particle with wave motion"""
    def __init__(self, x, y, scene):
        self.start_x = x
        self.start_y = y
        self.current_x = x
        self.current_y = y
        self.target_x = 0
        self.target_y = 0
        self.wave_phase = 0
        self.energy = 0.5 + (hash(str(x+y)) % 100) / 200  # Pseudo-random 0.5-1.0
        
        # Create visual particle
        size = 3 + self.energy * 4
        self.item = QGraphicsEllipseItem(x-size/2, y-size/2, size, size)
        
        # Quantum gradient
        gradient = QRadialGradient(0, 0, size/2)
        gradient.setColorAt(0, QColor("#ffffff"))
        gradient.setColorAt(0.5, QColor("#5dade2"))
        gradient.setColorAt(1, QColor("#3498db"))
        
        self.item.setBrush(QBrush(gradient))
        self.item.setPen(QPen(QColor("#5dade2"), 1))
        self.item.setOpacity(0)
        
        scene.addItem(self.item)
    
    def update_position(self, progress, time_ms):
        """Update particle position with wave motion"""
        # Wave motion
        wave_x = math.sin(time_ms * 0.003 + self.wave_phase) * 15
        wave_y = math.cos(time_ms * 0.003 + self.wave_phase) * 15
        
        # Interpolate to target
        self.current_x = self.start_x + (self.target_x - self.start_x) * progress + wave_x
        self.current_y = self.start_y + (self.target_y - self.start_y) * progress + wave_y
        
        # Update graphics item
        size = 3 + self.energy * 4
        self.item.setPos(self.current_x - size/2, self.current_y - size/2)
    
    def set_opacity(self, opacity):
        """Set particle opacity"""
        self.item.setOpacity(opacity * self.energy)


class QuantoniumIntro(QMainWindow):
    """Simplified QuantoniumOS Intro Animation"""
    
    animation_finished = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        # Constants
        self.phi = 1.618033988749895
        self.tau = 2 * math.pi
        
        # Setup window
        self.setWindowTitle("QuantoniumOS")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.showFullScreen()
        
        # Graphics setup
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        
        # Remove scrollbars and ensure proper fit
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setFrameStyle(0)
        self.view.setRenderHint(QPainter.Antialiasing, True)
        
        # Get screen geometry
        screen = QGuiApplication.primaryScreen().availableGeometry()
        self.scene.setSceneRect(0, 0, screen.width(), screen.height())
        
        # Set the view to show the entire scene
        self.view.setScene(self.scene)
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        
        self.setCentralWidget(self.view)
        
        # Screen center
        self.center_x = screen.width() / 2
        self.center_y = screen.height() / 2
        
        # Create components
        self.particles = []
        self.logo_elements = []
        self.text_elements = []
        
        self.create_particles()
        self.create_logo()
        self.create_text()
        
        # Animation state
        self.animation_step = 0
        self.start_time = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        print("🚀 QuantoniumOS Intro Ready")
    
    def create_particles(self):
        """Create quantum particles in spiral pattern"""
        particle_count = 36
        
        for i in range(particle_count):
            # Golden spiral
            angle = i * 137.508 * math.pi / 180  # Golden angle
            radius = math.sqrt(i) * 25
            
            x = self.center_x + radius * math.cos(angle)
            y = self.center_y + radius * math.sin(angle)
            
            # Keep on screen
            x = max(50, min(self.scene.width() - 50, x))
            y = max(50, min(self.scene.height() - 50, y))
            
            particle = QuantumParticle(x, y, self.scene)
            particle.wave_phase = i * 0.5
            
            # Target for convergence
            target_angle = angle * self.phi
            particle.target_x = self.center_x + 50 * math.cos(target_angle)
            particle.target_y = self.center_y + 50 * math.sin(target_angle)
            
            self.particles.append(particle)
    
    def create_logo(self):
        """Create Q logo"""
        # Outer circle
        outer_radius = 40
        outer_circle = QGraphicsEllipseItem(
            self.center_x - outer_radius,
            self.center_y - outer_radius,
            outer_radius * 2,
            outer_radius * 2
        )
        
        # Inner circle
        inner_radius = 25
        inner_circle = QGraphicsEllipseItem(
            self.center_x - inner_radius,
            self.center_y - inner_radius,
            inner_radius * 2,
            inner_radius * 2
        )
        
        # Q dash
        line_length = 25
        dash_x = self.center_x + 15
        dash_y = self.center_y + 15
        q_dash = QGraphicsLineItem(
            dash_x, dash_y,
            dash_x + line_length * 0.707,  # 45 degree angle
            dash_y + line_length * 0.707
        )
        
        # Style all elements
        pen = QPen(QColor("#3498db"), 3)
        pen.setCapStyle(Qt.RoundCap)
        
        for element in [outer_circle, inner_circle, q_dash]:
            element.setPen(pen)
            element.setOpacity(0)
            self.scene.addItem(element)
        
        # Only circles have brush
        for circle in [outer_circle, inner_circle]:
            circle.setBrush(QBrush(QColor(0, 0, 0, 0)))
        
        self.logo_elements = [outer_circle, inner_circle, q_dash]
    
    def create_text(self):
        """Create text elements"""
        # Title
        title = QGraphicsTextItem("QUANTONIUMOS")
        title_font = QFont("Arial", 24, QFont.Bold)
        title.setFont(title_font)
        title.setDefaultTextColor(QColor("#3498db"))
        
        # Center title below logo
        title_rect = title.boundingRect()
        title.setPos(
            self.center_x - title_rect.width()/2,
            self.center_y + 80
        )
        title.setOpacity(0)
        self.scene.addItem(title)
        
        # Subtitle
        subtitle = QGraphicsTextItem("Symbolic Quantum-Inspired Computing")
        subtitle_font = QFont("Arial", 14)
        subtitle.setFont(subtitle_font)
        subtitle.setDefaultTextColor(QColor("#85c1e9"))
        
        # Center subtitle below title
        subtitle_rect = subtitle.boundingRect()
        subtitle.setPos(
            self.center_x - subtitle_rect.width()/2,
            self.center_y + 120
        )
        subtitle.setOpacity(0)
        self.scene.addItem(subtitle)
        
        self.text_elements = [title, subtitle]
    
    def start_animation(self):
        """Start the animation sequence"""
        self.start_time = time.time() * 1000  # Convert to milliseconds
        self.animation_step = 0
        self.timer.start(33)  # ~30 FPS
        print("🌌 Animation Started!")
    
    def update_frame(self):
        """Update animation frame"""
        current_time = time.time() * 1000
        elapsed = current_time - self.start_time
        self.animation_step += 1
        
        # Phase 1: Particle fade-in (0-1500ms)
        if elapsed < 1500:
            for i, particle in enumerate(self.particles):
                delay = i * 40
                if elapsed > delay:
                    progress = min(1.0, (elapsed - delay) / 800)
                    particle.set_opacity(progress)
                    particle.update_position(0, current_time)
        
        # Phase 2: Particle convergence (1500-4000ms)
        elif elapsed < 4000:
            conv_progress = (elapsed - 1500) / 2500
            for particle in self.particles:
                particle.set_opacity(1.0 - conv_progress * 0.7)
                particle.update_position(conv_progress, current_time)
        
        # Phase 3: Logo formation (3500-5000ms)
        elif elapsed < 5000:
            logo_progress = (elapsed - 3500) / 1500
            for element in self.logo_elements:
                element.setOpacity(logo_progress)
        
        # Phase 4: Text appearance (4500-6500ms)
        elif elapsed < 6500:
            text_progress = (elapsed - 4500) / 2000
            for i, element in enumerate(self.text_elements):
                delay_factor = i * 0.3
                delayed_progress = max(0, text_progress - delay_factor)
                element.setOpacity(min(1.0, delayed_progress * 2))
        
        # Phase 5: Hold and finish (6500ms+)
        elif elapsed > 7500:
            self.timer.stop()
            print("✅ Animation Complete!")
            self.animation_finished.emit()
    
    def finish_animation(self):
        """Complete animation and signal finish"""
        self.animation_finished.emit()
    
    def keyPressEvent(self, event):
        """ESC to skip"""
        if event.key() == Qt.Key_Escape:
            self.timer.stop()
            self.finish_animation()
        super().keyPressEvent(event)


def main():
    """Test the intro animation"""
    app = QApplication(sys.argv)
    
    intro = QuantoniumIntro()
    intro.show()
    intro.start_animation()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
