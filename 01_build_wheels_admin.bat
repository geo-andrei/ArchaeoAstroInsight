@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo === Step 1/3: Locate QGIS Python 3.12 ===
set "QPY="
for /d %%D in ("C:\Program Files\QGIS*") do (
  if exist "%%D\apps\Python312\python.exe" set "QPY=%%D\apps\Python312\python.exe"
)
if not defined QPY if exist "C:\OSGeo4W\apps\Python312\python.exe" set "QPY=C:\OSGeo4W\apps\Python312\python.exe"
if not defined QPY (
  echo [ERROR] Could not find QGIS Python 3.12. Edit QPY below to your install if needed.
  echo Example: set "QPY=C:\Program Files\QGIS 3.40.10\apps\Python312\python.exe"
  pause & exit /b 1
)
echo Using: "%QPY%"
echo.

echo === Step 2/3: Make wheels\ folder ===
if not exist wheels mkdir wheels

echo === Step 3/3: Download all wheels (includes transitive deps) ===
"%QPY%" -m pip install --upgrade pip wheel setuptools
"%QPY%" -m pip download --only-binary=:all: -d wheels -r requirements.lock
REM Download Ultralytics wheel itself WITHOUT dependencies (to avoid GUI OpenCV)
"%QPY%" -m pip download --only-binary=:all: -d wheels "ultralytics>=8.3.0,<8.4" --no-deps
REM Download astrocult (TestPyPI) + multipledispatch, no deps (pure-python curvigram lib)
"%QPY%" -m pip download --no-deps --ignore-requires-python -d wheels --index-url https://test.pypi.org/simple/ astrocult
"%QPY%" -m pip download --no-deps -d wheels multipledispatch
REM skyfield (+ jplephem, sgp4) — needed by astrocult's declination curvigrams
"%QPY%" -m pip download --no-deps -d wheels skyfield jplephem sgp4

echo.
echo [OK] Wheels cached in: %CD%\wheels
echo If any package is missing, re-run this script while connected to the Internet.
echo.
dir /b wheels\*tensorflow* wheels\*torch* wheels\*opencv* 2>nul
echo.
pause
endlocal
