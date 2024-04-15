@echo off
REM Find all running Python processes
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq python.exe" /nh') do (
    REM Terminate each Python process forcefully
    taskkill /pid %%i /f
)