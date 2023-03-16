import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "packages": ["librosa"],
    "zip_include_packages": ["PySide6"],
    "include_files": ["./D1.pt", "./icon.png"],
    "include_msvcr": True,
}

# base="Win32GUI" should be used only for Windows GUI app
base = "Win32GUI" if sys.platform == "win32" else None

setup(
    name="Kunstzeal",
    version="1.0",
    description="Kunstzeal Deployed GUI",
    options={"build_exe": build_exe_options},
    executables=[Executable("main.py", base=base, target_name="Kunstzeal")],   
)