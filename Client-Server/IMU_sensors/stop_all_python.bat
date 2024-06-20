@echo off

echo Attempting to stop all Python scripts...

REM Use Task Manager to end all Python processes
taskkill /F /IM python3.8.exe /T
taskkill /F /IM python /T

echo All Python scripts stopped.
