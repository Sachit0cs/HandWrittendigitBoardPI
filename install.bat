@echo off
echo ============================================
echo  Handwriting Digit Board — Dependency Setup
echo ============================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo [1/3] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [2/3] Installing packages...
pip install opencv-python numpy tensorflow Flask Pillow

echo.
echo [3/3] All done!
echo.
echo Next steps:
echo   1. python train_model.py      (trains and saves the CNN - ~2 minutes)
echo   2. python main.py             (starts the live digit board)
echo   3. python flask_dashboard.py  (optional web dashboard)
echo.
pause
