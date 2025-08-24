#!/usr/bin/env python3
"""
QuantoniumOS - Unified Design System
====================================
Centralized styling and proportions for all QuantoniumOS applications.
Separates design from functionality for easy global modifications.
"""

import math

from PyQt5.QtCore import QSize
from PyQt5.QtGui import QColor, QFont


class QuantoniumDesignSystem:
    """Centralized design system with calculated proportions"""

    def __init__(self, screen_width=1920, screen_height=1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self._calculate_proportions()
        self._define_colors()
        self._define_fonts()
        self._define_styles()

    def _calculate_proportions(self):
        """Calculate optimal proportions based on screen size"""
        # Base unit calculation (golden ratio proportions)
        self.base_unit = min(self.screen_width, self.screen_height) / 80

        # Dock proportions
        self.dock_collapsed_width = self.screen_width * 0.025  # 2.5% of screen width
        self.dock_expanded_width = self.screen_width * 0.28  # 28% of screen width
        self.dock_height = self.screen_height * 0.85  # 85% of screen height
        self.dock_tab_height = self.screen_height * 0.12  # 12% of screen height

        # Oval Dock Tab
        self.dock_tab_radius = self.dock_tab_height / 2  # Radius for D-shape

        # Circular arrangement
        self.app_circle_radius = self.dock_expanded_width * 0.35  # 35% of expanded dock
        self.app_icon_size = int(self.base_unit * 1.2)  # 1.2 base units
        self.app_spacing_angle = 180 / 10  # For 9 apps in a semi-circle

        # Central Q logo
        self.q_logo_size = int(self.base_unit * 4.5)  # 4.5 base units

        # Clock positioning
        self.clock_margin = self.base_unit * 0.8

        # App window sizes
        self.app_window_width = int(self.screen_width * 0.7)  # 70% of screen
        self.app_window_height = int(self.screen_height * 0.8)  # 80% of screen

    def _define_colors(self):
        """Define the color palette"""
        # Primary colors (cream theme)
        self.colors = {
            "primary_bg": "#f0ead6",  # Main cream background
            "secondary_bg": "#f8f6f0",  # Lighter cream for containers
            "text_primary": "#2d2d2d",  # Dark gray text
            "text_secondary": "#555555",  # Medium gray text
            "accent_blue": "#0078d4",  # Microsoft blue
            "accent_blue_hover": "#106ebe",  # Darker blue for hover
            "logo_gray": "#6b6b6b",  # Gray for Q logo
            "icon_black": "#000000",  # Black for icons
            "border_light": "#c0c0c0",  # Light gray borders
            "dock_overlay": "rgba(200, 200, 200, 0.4)",  # Dock background
            "app_name_bg": "rgba(248, 246, 240, 0.9)",  # App name background
        }

    def _define_fonts(self):
        """Define font specifications"""
        self.fonts = {
            "primary_family": "Segoe UI",
            "monospace_family": "Consolas",
            "q_logo_size": self.q_logo_size,
            "clock_size": int(self.base_unit * 0.6),
            "app_name_size": int(self.base_unit * 0.4),
            "button_size": int(self.base_unit * 0.45),
            "content_size": int(self.base_unit * 0.4),
        }

    def _define_styles(self):
        """Define QSS stylesheets for different components"""
        self.styles = {
            "main_window": f"""
                QMainWindow {{
                    background-color: {self.colors['primary_bg']};
                    color: {self.colors['text_primary']};
                    font-family: "{self.fonts['primary_family']}";
                    font-size: {self.fonts['content_size']}pt;
                }}
            """,
            "q_logo": f"""
                color: {self.colors['logo_gray']};
                font: bold {self.fonts['q_logo_size']}pt "{self.fonts['primary_family']}";
                background: transparent;
            """,
            "clock": f"""
                color: {self.colors['text_primary']};
                font: bold {self.fonts['clock_size']}pt "{self.fonts['primary_family']}";
                background: transparent;
            """,
            "app_icon": f"""
                color: {self.colors['icon_black']};
                background: transparent;
                border: none;
            """,
            "app_name": f"""
                color: {self.colors['text_primary']};
                font: bold {self.fonts['app_name_size']}pt "{self.fonts['primary_family']}";
                background: {self.colors['app_name_bg']};
                border-radius: 4px;
                padding: 2px 6px;
            """,
            "button_primary": f"""
                QPushButton {{
                    background-color: {self.colors['accent_blue']};
                    border: 1px solid {self.colors['accent_blue']};
                    border-radius: 4px;
                    padding: 8px 16px;
                    color: white;
                    font: {self.fonts['button_size']}pt "{self.fonts['primary_family']}";
                    font-weight: normal;
                    min-width: 80px;
                }}
                QPushButton:hover {{
                    background-color: {self.colors['accent_blue_hover']};
                }}
                QPushButton:pressed {{
                    background-color: #005a9e;
                }}
                QPushButton:disabled {{
                    background-color: #cccccc;
                    border: 1px solid #999999;
                    color: #666666;
                }}
            """,
            "container": f"""
                QWidget {{
                    background-color: {self.colors['secondary_bg']};
                    border: 1px solid {self.colors['border_light']};
                    border-radius: 8px;
                    padding: 10px;
                }}
            """,
            "input_field": f"""
                QLineEdit {{
                    background-color: white;
                    border: 1px solid {self.colors['border_light']};
                    border-radius: 4px;
                    padding: 6px;
                    color: {self.colors['text_primary']};
                    font-size: {self.fonts['content_size']}pt;
                }}
                QLineEdit:focus {{
                    border: 2px solid {self.colors['accent_blue']};
                }}
            """,
            "text_area": f"""
                QTextEdit {{
                    background-color: white;
                    border: 1px solid {self.colors['border_light']};
                    border-radius: 4px;
                    color: {self.colors['text_primary']};
                    font-size: {self.fonts['content_size']}pt;
                    padding: 8px;
                }}
            """,
            "group_box": f"""
                QGroupBox {{
                    border: 1px solid {self.colors['border_light']};
                    border-radius: 6px;
                    padding: 10px;
                    margin-top: 10px;
                    font-weight: normal;
                    background-color: {self.colors['secondary_bg']};
                }}
                QGroupBox::title {{
                    color: {self.colors['text_primary']};
                    padding: 5px;
                    font-weight: bold;
                }}
            """,
            "scroll_area": f"""
                QScrollArea {{
                    background: transparent;
                    border: none;
                }}
                QScrollBar:vertical {{
                    background: #e0e0e0;
                    width: 12px;
                    border-radius: 6px;
                }}
                QScrollBar::handle:vertical {{
                    background: {self.colors['accent_blue']};
                    border-radius: 6px;
                    min-height: 20px;
                }}
                QScrollBar::handle:vertical:hover {{
                    background: {self.colors['accent_blue_hover']};
                }}
            """,
        }

    def get_dock_geometry(self, expanded=False):
        """Get dock geometry based on state"""
        if expanded:
            return {
                "width": self.dock_expanded_width,
                "height": self.dock_height,
                "x": 0,
                "y": (self.screen_height - self.dock_height) / 2,
            }
        else:
            return {
                "width": self.dock_collapsed_width,
                "height": self.dock_tab_height,
                "x": 0,
                "y": (self.screen_height - self.dock_tab_height) / 2,
            }

    def get_app_positions(self):
        """Calculate positions for apps in semi-circular arrangement"""
        positions = []
        center_x = self.dock_expanded_width * 0.5
        center_y = self.screen_height * 0.5

        # Create semi-circular arrangement (180 degrees)
        for i in range(9):  # 9 apps
            # Calculate angle in the 180-degree arc (from -90 to 90 degrees)
            angle = math.radians(i * self.app_spacing_angle - 90)

            # Position on the right half of the circle only
            x = center_x + self.app_circle_radius * math.cos(angle)
            y = center_y + self.app_circle_radius * math.sin(angle)
            positions.append({"x": x, "y": y})

        return positions

    def get_q_logo_position(self):
        """Get center position for Q logo"""
        return {
            "x": (self.screen_width - self.q_logo_size * 1.5)
            / 2,  # Approximate text width
            "y": (self.screen_height - self.q_logo_size) / 2,
        }

    def get_clock_position(self):
        """Get position for clock in top-right"""
        return {
            "x": self.screen_width - 250,  # Account for clock text width
            "y": self.clock_margin,
        }

    def get_app_window_geometry(self):
        """Get standard app window geometry"""
        return {
            "width": self.app_window_width,
            "height": self.app_window_height,
            "x": (self.screen_width - self.app_window_width) / 2,
            "y": (self.screen_height - self.app_window_height) / 2,
        }

    def apply_style(self, widget, style_name):
        """Apply a style to a widget"""
        if style_name in self.styles:
            widget.setStyleSheet(self.styles[style_name])
        else:
            print(f"⚠️ Style '{style_name}' not found in design system")

    def get_unified_app_stylesheet(self):
        """Get complete stylesheet for applications"""
        return f"""
            /* QuantoniumOS Unified Design System */
            {self.styles['main_window']}
            
            /* Buttons */
            {self.styles['button_primary']}
            
            /* Input Fields */
            {self.styles['input_field']}
            
            /* Text Areas */
            {self.styles['text_area']}
            
            /* Group Boxes */
            {self.styles['group_box']}
            
            /* Scroll Areas */
            {self.styles['scroll_area']}
            
            /* Lists */
            QListWidget {{
                background-color: white;
                border: 1px solid {self.colors['border_light']};
                border-radius: 4px;
                color: {self.colors['text_primary']};
                font-size: {self.fonts['content_size']}pt;
            }}
            QListWidget::item {{
                padding: 8px;
                border-bottom: 1px solid #f0f0f0;
            }}
            QListWidget::item:selected {{
                background-color: {self.colors['accent_blue']};
                color: white;
            }}
            QListWidget::item:hover {{
                background-color: #f0f8ff;
            }}
            
            /* Tables */
            QTableWidget {{
                background-color: white;
                border: 1px solid {self.colors['border_light']};
                border-radius: 4px;
                gridline-color: #f0f0f0;
            }}
            QTableWidget::item {{
                padding: 8px;
                color: {self.colors['text_primary']};
            }}
            QTableWidget::item:selected {{
                background-color: {self.colors['accent_blue']};
                color: white;
            }}
            QHeaderView::section {{
                background-color: {self.colors['secondary_bg']};
                padding: 8px;
                border: 1px solid {self.colors['border_light']};
                font-weight: bold;
            }}
            
            /* Tabs */
            QTabWidget::pane {{
                border: 1px solid {self.colors['border_light']};
                background-color: white;
                border-radius: 4px;
            }}
            QTabBar::tab {{
                background-color: #e6e6e6;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                color: {self.colors['text_primary']};
            }}
            QTabBar::tab:selected {{
                background-color: {self.colors['accent_blue']};
                color: white;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: #d0d0d0;
            }}
            
            /* Progress Bars */
            QProgressBar {{
                background-color: #f0f0f0;
                border: 1px solid {self.colors['border_light']};
                border-radius: 4px;
                text-align: center;
                color: {self.colors['text_primary']};
                height: 20px;
            }}
            QProgressBar::chunk {{
                background-color: {self.colors['accent_blue']};
                border-radius: 3px;
            }}
            
            /* Status Bar */
            QStatusBar {{
                background-color: {self.colors['secondary_bg']};
                border-top: 1px solid {self.colors['border_light']};
                color: {self.colors['text_secondary']};
            }}
            
            /* Menu Bar */
            QMenuBar {{
                background-color: {self.colors['secondary_bg']};
                border-bottom: 1px solid {self.colors['border_light']};
                color: {self.colors['text_primary']};
            }}
            QMenuBar::item {{
                padding: 8px 12px;
            }}
            QMenuBar::item:selected {{
                background-color: {self.colors['accent_blue']};
                color: white;
            }}
        """


# Global design system instance
design_system = None


def get_design_system(screen_width=None, screen_height=None):
    """Get or create the global design system instance"""
    global design_system
    if design_system is None or (screen_width and screen_height):
        if screen_width and screen_height:
            design_system = QuantoniumDesignSystem(screen_width, screen_height)
        else:
            design_system = QuantoniumDesignSystem()
    return design_system


def apply_unified_style(widget):
    """Apply unified styling to any widget"""
    ds = get_design_system()
    widget.setStyleSheet(ds.get_unified_app_stylesheet())


def get_app_geometry():
    """Get standard geometry for app windows"""
    ds = get_design_system()
    return ds.get_app_window_geometry()
