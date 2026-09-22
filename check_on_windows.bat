@echo off
setlocal
cd /d "%~dp0"
echo.
echo   Reproduction check for delayed-information-cross-section
echo   -------------------------------------------------------
echo   This reruns all the analysis scripts and compares every line
echo   against the stored reference output. It takes about an hour.
echo   Leave this window open; it will tell you when it is done.
echo.
where python >nul 2>&1
if errorlevel 1 (
  echo   Python was not found.
  echo.
  echo   Open Anaconda Prompt from the Start menu, type  cd  followed by a
  echo   space, drag this folder onto the window, press Enter, then type:
  echo.
  echo       python run_all.py --check
  echo.
  pause
  exit /b 1
)
python run_all.py --check > check_windows.log 2>&1
echo.
echo   Finished. The full transcript is in check_windows.log
echo.
findstr /C:"reproduced the reference output exactly" check_windows.log
findstr /C:"did not reproduce" check_windows.log
findstr /C:"DIFFERS" check_windows.log
echo.
pause
