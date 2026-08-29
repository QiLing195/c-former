@echo off
chcp 65001 >nul
setlocal
cd /d "E:\deepseek\c-former"

echo ============================================================
echo  C-Former re-push script
echo  Order: clean - fix commit - TEST - push (test-gated)
echo ============================================================

echo.
echo  [Step 1/5] remove nested clone c-former-v65-line
echo  (V6.5 mainline backup is kept at E:\oprncode\c-former)
if exist "E:\deepseek\c-former\c-former-v65-line" (
  rmdir /s /q "E:\deepseek\c-former\c-former-v65-line"
  echo    removed
) else (
  echo    not found - skip
)

echo.
echo  [Step 2/5] fix typo commit "ling" - amend message
git commit --amend -m "push.bat: rewrite as ASCII to fix GBK garbling ('ell' error), keep chcp 65001" || goto :err

echo.
echo  [Step 3/5] show current remote URL (must match new empty repo)
git remote -v

echo.
echo  [Step 4/5] run test suite (fail = abort, no push)
D:\conda\envs\cformer-gpu\python.exe -m pytest tests/ -q || goto :testfail

echo.
echo  [Step 5/5] push to GitHub
echo  IMPORTANT: create the EMPTY repo first at github.com/new
echo  (do NOT tick "Add a README file").
pause
git add -A || goto :err
git commit -m "README: session-line V6.3 version overview (real-data results, recursion, cross-domain, TTT)" || echo  (nothing to commit - ok)
git push -u origin master || goto :err
git push origin --tags || echo  (tag push skipped/failed - not fatal)

echo.
echo  DONE. Check https://github.com/QiLing195/c-former
pause
exit /b 0

:testfail
echo.
echo  [ABORT] tests failed - NOT pushed. Fix tests, rerun.
pause
exit /b 1

:err
echo.
echo  [ERROR] stopped. Fix the problem shown above, rerun.
pause
exit /b 1
