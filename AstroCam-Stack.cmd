@echo off
powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0AstroCam-Stack\AstroCam-Stack.ps1" %*
pause
