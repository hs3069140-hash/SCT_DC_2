@echo off
echo ========================================================
echo   SkillCraft Technology - Task 02: Titanic EDA Runner
echo ========================================================
echo.

echo [1/3] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.9+ from https://python.org and try again.
    pause
    exit /b
)

echo [2/3] Installing/verifying requirements...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [WARNING] Encountered an issue installing packages. Trying to run anyway...
)

echo.
echo [3/3] Running Titanic Exploratory Data Analysis...
echo.
python titanic_eda.py

echo.
echo ========================================================
echo   Done! All charts generated in the 'charts' directory.
echo ========================================================
echo Opening charts folder...
start "" "%~dp0charts"
pause
