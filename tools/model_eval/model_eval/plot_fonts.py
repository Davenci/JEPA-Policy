"""Register available CJK fonts for trajectory plots."""

from pathlib import Path


def register_cjk_fonts(font_manager) -> list[str]:
    """Load installed CJK font files into Matplotlib and return their families."""
    font_paths = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    )
    registered_paths = {
        Path(font.fname).resolve() for font in font_manager.fontManager.ttflist
    }
    families = []
    for font_path in font_paths:
        if not font_path.is_file():
            continue
        if font_path.resolve() not in registered_paths:
            font_manager.fontManager.addfont(str(font_path))
            registered_paths.add(font_path.resolve())
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
        if family not in families:
            families.append(family)
    return families
