@echo off
rem Stock Cone: start the local server and open the page in the default browser.
rem Which Python: set STOCK_CONE_PYTHON to a python.exe path to choose one. Otherwise
rem the Windows py launcher (py -3) is used, then python on PATH.
cd /d "%~dp0"
rem open the page ~2 seconds from now, once the server is listening
start "" /min cmd /c "ping -n 3 127.0.0.1 >nul & start http://127.0.0.1:8642"
if defined STOCK_CONE_PYTHON goto envpy
where py >nul 2>nul
if errorlevel 1 goto pathpy
py -3 server.py
goto done
:envpy
"%STOCK_CONE_PYTHON%" server.py
goto done
:pathpy
python server.py
:done
pause
