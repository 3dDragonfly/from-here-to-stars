@echo off
powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0AstroCam-Capture.ps1" %*
pause
