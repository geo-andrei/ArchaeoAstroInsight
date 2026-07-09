@echo off
setlocal
cd /d "%~dp0"

REM Find QGIS Python
set "QPY="
for /d %%D in ("C:\Program Files\QGIS*") do (
  if exist "%%D\apps\Python312\python.exe" set "QPY=%%D\apps\Python312\python.exe"
)
if not defined QPY if exist "C:\OSGeo4W\apps\Python312\python.exe" set "QPY=C:\OSGeo4W\apps\Python312\python.exe"
if not defined QPY (
  echo [ERROR] QGIS Python not found.
  pause & exit /b 1
)

REM Run the standalone verifier (works in cmd; no bash heredoc needed)
"%QPY%" "%~dp0verify_install.py"
if errorlevel 1 (
  echo.
  echo [FAILED] Some dependencies failed to import. See the messages above.
) else (
  echo.
  echo [OK] All dependencies import correctly.
)

pause
endlocal
