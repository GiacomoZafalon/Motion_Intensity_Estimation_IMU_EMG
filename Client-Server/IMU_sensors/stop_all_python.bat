@echo off

echo Attempting to stop all Python scripts...

REM Use Task Manager to end all Python processes
taskkill /F /IM python.exe /T

echo All Python scripts stopped.
