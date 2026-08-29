@echo off
rem C-Former daily push script (ASCII-only to avoid cmd encoding issues)
rem Usage: double-click, or run .\push.bat in PowerShell/cmd
rem Commit message may be typed in Chinese (console is set to UTF-8 below).

chcp 65001 >nul

cd /d E:\deepseek\c-former

if not exist .git (
    echo [ERROR] not a git repo - check path E:\deepseek\c-former
    pause
    exit /b 1
)

echo current dir: %cd%
echo.

set /p MSG=Commit message (Enter = update): 
if "%MSG%"=="" set MSG=update

echo.
echo message: %MSG%
echo.

git add -A
git commit -m "%MSG%"
git push

echo.
echo === DONE, recent commits ===
git log --oneline -3

echo.
echo If red errors appear above, send them to me.
pause
