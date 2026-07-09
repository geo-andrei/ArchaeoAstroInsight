@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo === ArchaeoAstroInsight: global (admin) install into QGIS Python ===

REM Ensure we're elevated
net session >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Please run this from an Administrator Command Prompt.
  pause & exit /b 1
)

REM Find QGIS Python 3.12
set "QPY="
for /d %%D in ("C:\Program Files\QGIS*") do (
  if exist "%%D\apps\Python312\python.exe" set "QPY=%%D\apps\Python312\python.exe"
)
if not defined QPY if exist "C:\OSGeo4W\apps\Python312\python.exe" set "QPY=C:\OSGeo4W\apps\Python312\python.exe"
if not defined QPY (
  echo [ERROR] QGIS Python 3.12 not found. Edit QPY in this file if needed.
  pause & exit /b 1
)
echo Using QGIS Python: "%QPY%"
echo.

REM Sanity: wheels + lock must exist
if not exist "wheels" (
  echo [ERROR] wheels\ folder not found. Run 01_build_wheels_admin.bat first.
  pause & exit /b 1
)
if not exist "requirements.lock" (
  echo [ERROR] requirements.lock not found next to this script.
  pause & exit /b 1
)

echo Upgrading pip tooling...
"%QPY%" -m pip install --upgrade pip wheel setuptools || goto :err

echo Removing GUI OpenCV if present (harmless if not)...
"%QPY%" -m pip uninstall -y opencv-python opencv-contrib-python >nul 2>&1

echo Installing all deps from local wheels (offline)...
"%QPY%" -m pip install --no-index --find-links=.\wheels -r requirements.lock || goto :err

echo Installing Ultralytics WITHOUT deps (avoids GUI OpenCV)...
"%QPY%" -m pip install --no-deps --no-index --find-links=.\wheels "ultralytics>=8.3.0,<8.4"

echo Installing astrocult (+ multipledispatch, skyfield) WITHOUT deps (curvigram library)...
"%QPY%" -m pip install --no-deps --ignore-requires-python --no-index --find-links=.\wheels astrocult multipledispatch skyfield jplephem sgp4

echo.
echo === Copy/refresh plugin into user profile ===
set "QPROF=%APPDATA%\QGIS\QGIS3\profiles\default"
set "PLUGINS=%QPROF%\python\plugins"
set "PLUGIN_NAME=ArchaeoAstroInsight"
set "TARGET_PLUGIN=%PLUGINS%\%PLUGIN_NAME%"
REM Source is this script's own folder (the plugin root)
set "SOURCE=%CD%"

if not exist "%PLUGINS%" mkdir "%PLUGINS%" >nul 2>&1

REM Remove a stale OLD-named install -- but never delete the folder we run from
if exist "%PLUGINS%\arche_detection5" if /i not "%SOURCE%"=="%PLUGINS%\arche_detection5" rmdir /s /q "%PLUGINS%\arche_detection5"

REM If this script is already running from its target profile folder, keep it in
REM place: deleting/copying onto ourselves would wipe the plugin. Just skip copy.
if /i "%SOURCE%"=="%TARGET_PLUGIN%" (
  echo Plugin already in the QGIS profile -- installed in place, skipping copy.
  goto :deployed
)

if exist "%TARGET_PLUGIN%" rmdir /s /q "%TARGET_PLUGIN%"
echo Copying plugin from "%SOURCE%"
echo                  to   "%TARGET_PLUGIN%"
REM Copy plugin, skipping dev/build artifacts that must not ship to QGIS
robocopy "%SOURCE%" "%TARGET_PLUGIN%" /E /XD ".venv" "__pycache__" "wheels" /XF "*.pyc" >nul
if errorlevel 8 (
  echo [ERROR] Plugin copy failed.
  goto :err
)
echo Plugin copied successfully.

:deployed

echo.
echo [DONE] Global deps installed to QGIS site-packages.
echo       Plugin (if present) was copied to your profile.
echo       Start QGIS -> Plugins -> enable "ArchaeoAstroInsight".
echo.
pause
exit /b 0

:err
echo.
echo [FAILED] See the error above.
echo Make sure this window was opened as Administrator.
pause
exit /b 1
