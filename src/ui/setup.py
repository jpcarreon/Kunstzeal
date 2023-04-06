import sys
from cx_Freeze import setup, Executable

try:
    from cx_Freeze.hooks import get_qt_plugins_paths
except ImportError:
    get_qt_plugins_paths = None

include_files = ["./D1.pt", "./icon.ico"]
if get_qt_plugins_paths:
    for plugin_name in (
        "wayland-decoration-client",
        "wayland-graphics-integration-client",
        "wayland-shell-integration",
    ):
        include_files += get_qt_plugins_paths("PySide6", plugin_name)

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "packages": ["librosa"],
    "zip_include_packages": ["PySide6"],
    "include_files": include_files,
    "include_msvcr": True,
}

# base="Win32GUI" should be used only for Windows GUI app
base = "Win32GUI" if sys.platform == "win32" else None

setup(
    name="Kunstzeal",
    version="1.2.1",
    description="Kunstzeal",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            "main.py", 
            base=base, 
            icon="icon.ico",
            target_name="Kunstzeal"
    )],   
)