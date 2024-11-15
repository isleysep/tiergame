@echo off
setlocal enabledelayedexpansion

:: Initialize the virtual environment and install requirements
echo Initializing virtual environment...
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt

:: Provide a simple GUI for user selection
:menu
cls
echo ===============================
echo    Select Phase to Run:
echo ===============================
echo 1. Phase One (Requires YouTube Playlist URL)
echo 2. Phase Two (Checks for input folder)
echo 3. Check for Updates
echo 4. Exit
echo ===============================
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" goto phase_one
if "%choice%"=="2" goto phase_two
if "%choice%"=="3" goto check_updates
if "%choice%"=="4" goto end
echo Invalid choice, please try again.
pause
goto menu

:phase_one
set /p url="Enter the YouTube Playlist URL: "
echo Running phase_one.py with the provided URL...
python phase_one.py "%url%"
pause
goto menu

:phase_two
if exist "input" (
    echo Input folder exists. Running phase_two.py...
    python phase_two.py
) else (
    echo Error: input folder does not exist. Please create the input folder and try again.
)
pause
goto menu

:check_updates
echo Checking for updates from the Git repository...
git pull https://github.com/isleysep/tiergame
pause
goto menu

:end
echo Exiting...
deactivate
exit
