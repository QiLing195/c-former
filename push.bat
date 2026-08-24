@echo off
chcp 65001 >nul
rem ============================================================
rem  C-Former 每次推送脚�?rem  用法：双击本文件，或�?PowerShell 里执�?.\push.bat
rem  会先问你「这次改了什么」，输入后自�?add + commit + push
rem ============================================================

cd /d E:\oprncode\c-former

if not exist .git (
    echo [ERROR] 当前目录不是 git 仓库，请确认项目�?E:\oprncode\c-former
    pause
    exit /b 1
)

echo 当前目录: %cd%
echo.

set /p MSG=请输入这次改了什么（直接回车 = update�? 
if "%MSG%"=="" set MSG=update

echo.
echo 提交信息: %MSG%
echo.

git add -A
git commit -m "%MSG%"
git push

echo.
echo === 完成，最近三次提�?===
git log --oneline -3

echo.
echo 如果上面出现红色报错，把报错截图发出来�?pause
