@echo off
rem Lance l'outil Avatar LSFB (Windows) : double-cliquer. Laisser la fenetre ouverte pendant l'utilisation.
chcp 65001 >nul
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
cd /d "%~dp0..\..\.."
where uv >nul 2>nul || (echo L'outil n'est pas installe : double-cliquez d'abord sur Installer.bat. & pause & exit /b 1)
uv run --quiet --no-project --python 3.12 --with numpy python src\sign_language_3d_avatar\local_website\serveur.py --ouvrir
pause
