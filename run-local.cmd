@echo off
rem Double-click to run Sendy on this laptop for phone testing over Wi-Fi.
rem Leave this window open; the app stops when you close it or the laptop sleeps.
cd /d "%~dp0"
set SENDY_DEV=1
echo.
echo  Sendy starting. On your phone (same Wi-Fi), open the "Network URL" printed below.
echo  Close this window to stop.
echo.
venv\Scripts\streamlit.exe run app.py --server.headless true
