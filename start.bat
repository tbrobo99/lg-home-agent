@echo off
chcp 65001 >nul
cd /d %~dp0
echo Starting Proactive Senior Care ^& Comfort Agent...
python run.py
if errorlevel 1 (
  echo.
  echo Failed to start. Make sure Python 3.9+ is installed and on PATH.
  pause
)
