"""
QUANTONIUM DESIGN MANUAL
========================
Comprehensive guide to the QuantoniumOS design system.
"""


class Colors:
    """Color palette for QuantoniumOS"""

    PRIMARY = "#3C55F2"
    SECONDARY = "#6A7FE3"
    SUCCESS = "#26E07F"
    WARNING = "#FFB74D"
    ERROR = "#FF5252"
    INFO = "#29B6F6"
    BACKGROUND = "#121212"
    SURFACE = "#1E1E1E"
    TEXT_PRIMARY = "#FFFFFF"
    TEXT_SECONDARY = "#B3B3B3"

    @classmethod
    def as_dict(cls):
        """Return all colors as a dictionary"""
        return {
            "primary": cls.PRIMARY,
            "secondary": cls.SECONDARY,
            "success": cls.SUCCESS,
            "warning": cls.WARNING,
            "error": cls.ERROR,
            "info": cls.INFO,
            "background": cls.BACKGROUND,
            "surface": cls.SURFACE,
            "text_primary": cls.TEXT_PRIMARY,
            "text_secondary": cls.TEXT_SECONDARY,
        }

    @classmethod
    def as_list(cls):
        """Return all colors as a list of tuples"""
        return list(cls.as_dict().items())


class Typography:
    """Typography specifications for QuantoniumOS"""

    FONT_FAMILY = "Roboto, sans-serif"
    HEADING_1 = {"size": "32px", "weight": "700", "line_height": "40px"}
    HEADING_2 = {"size": "24px", "weight": "700", "line_height": "32px"}
    HEADING_3 = {"size": "20px", "weight": "600", "line_height": "28px"}
    BODY_1 = {"size": "16px", "weight": "400", "line_height": "24px"}
    BODY_2 = {"size": "14px", "weight": "400", "line_height": "20px"}
    CAPTION = {"size": "12px", "weight": "400", "line_height": "16px"}
    BUTTON = {"size": "14px", "weight": "500", "line_height": "20px"}

    @classmethod
    def as_dict(cls):
        """Return typography specifications as a dictionary"""
        return {
            "font_family": cls.FONT_FAMILY,
            "heading_1": cls.HEADING_1,
            "heading_2": cls.HEADING_2,
            "heading_3": cls.HEADING_3,
            "body_1": cls.BODY_1,
            "body_2": cls.BODY_2,
            "caption": cls.CAPTION,
            "button": cls.BUTTON,
        }


class Spacing:
    """Spacing system for QuantoniumOS"""

    UNIT = 8
    XS = UNIT / 2  # 4px
    SM = UNIT  # 8px
    MD = UNIT * 2  # 16px
    LG = UNIT * 3  # 24px
    XL = UNIT * 4  # 32px
    XXL = UNIT * 6  # 48px

    @classmethod
    def as_dict(cls):
        """Return spacing system as a dictionary"""
        return {
            "unit": cls.UNIT,
            "xs": cls.XS,
            "sm": cls.SM,
            "md": cls.MD,
            "lg": cls.LG,
            "xl": cls.XL,
            "xxl": cls.XXL,
        }


class Shadows:
    """Shadow system for QuantoniumOS"""

    NONE = "none"
    SM = "0 2px 4px rgba(0, 0, 0, 0.1)"
    MD = "0 4px 8px rgba(0, 0, 0, 0.12)"
    LG = "0 8px 16px rgba(0, 0, 0, 0.14)"
    XL = "0 16px 24px rgba(0, 0, 0, 0.16)"

    @classmethod
    def as_dict(cls):
        """Return shadow system as a dictionary"""
        return {
            "none": cls.NONE,
            "sm": cls.SM,
            "md": cls.MD,
            "lg": cls.LG,
            "xl": cls.XL,
        }


class BorderRadius:
    """Border radius system for QuantoniumOS"""

    NONE = "0px"
    SM = "4px"
    MD = "8px"
    LG = "16px"
    FULL = "9999px"

    @classmethod
    def as_dict(cls):
        """Return border radius system as a dictionary"""
        return {
            "none": cls.NONE,
            "sm": cls.SM,
            "md": cls.MD,
            "lg": cls.LG,
            "full": cls.FULL,
        }


class DesignSystem:
    """Complete design system for QuantoniumOS"""

    def __init__(self):
        """Initialize the design system"""
        self.colors = Colors()
        self.typography = Typography()
        self.spacing = Spacing()
        self.shadows = Shadows()
        self.border_radius = BorderRadius()

    def get_all(self):
        """Get the complete design system"""
        return {
            "colors": Colors.as_dict(),
            "typography": Typography.as_dict(),
            "spacing": Spacing.as_dict(),
            "shadows": Shadows.as_dict(),
            "border_radius": BorderRadius.as_dict(),
        }


def get_design_system():
    """Get the QuantoniumOS design system"""
    return DesignSystem()


def apply_theme(component, theme="dark"):
    """Apply a theme to a component"""
    themes = {
        "dark": {
            "background": Colors.BACKGROUND,
            "foreground": Colors.TEXT_PRIMARY,
            "accent": Colors.PRIMARY,
        },
        "light": {
            "background": "#F5F5F5",
            "foreground": "#212121",
            "accent": Colors.PRIMARY,
        },
        "quantum": {
            "background": "#0A0E21",
            "foreground": "#E6E6FA",
            "accent": "#7A3CFF",
        },
    }

    if theme not in themes:
        theme = "dark"

    component.theme = themes[theme]
    return component


if __name__ == "__main__":
    # Print the design system when run directly
    design = get_design_system()
    import json

    print(json.dumps(design.get_all(), indent=2))
