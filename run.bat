@echo off
rem Stock Cone: start the local server and open the page in the default browser.
cd /d "%~dp0"
rem open the page ~2 seconds from now, once the server is listening
start "" /min cmd /c "ping -n 3 127.0.0.1 >nul & start http://127.0.0.1:8642"
"python" server.py
pause
