@echo off
rem Installe ce dont l'outil Avatar LSFB a besoin (Windows) : double-cliquer, une seule fois.
chcp 65001 >nul
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
cd /d "%~dp0..\..\.."
echo Installation de l'outil Avatar LSFB...
where uv >nul 2>nul || powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
uv run --quiet --no-project --python 3.12 --with numpy python -c "import numpy" || (echo Echec de la preparation de Python. & pause & exit /b 1)
if exist "C:\Program Files\Blender Foundation" (echo - Blender : trouve.) else (echo - ATTENTION : Blender introuvable. Installez-le depuis blender.org, voir la notice.)
echo.
echo Installation terminee. Double-cliquez ensuite sur "Avatar LSFB.bat".
pause
