@echo off
REM Start the skyfind workbench from a source checkout.
REM Uses the project virtualenv when one is present, otherwise falls back to
REM `uv run`, which resolves the [gui] extra on the fly.
REM Arguments are passed through to `skyfind gui` (a folder of clips, or one
REM clip), e.g.  gui.cmd my-clips
setlocal
pushd "%~dp0"

REM this machine's settings (SKYFIND_DATA and the like), from the
REM untracked gui.local.cmd when there is one
if exist "gui.local.cmd" call "gui.local.cmd"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m skyfind gui %*
    goto :done
)

where uv >nul 2>&1
if errorlevel 1 goto :noenv
uv run --extra gui python -m skyfind gui %*
goto :done

:noenv
echo No .venv and no uv on PATH - cannot start the skyfind workbench.
echo Create the environment first, e.g.:  uv sync --extra gui
popd
exit /b 1

:done
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" (
    echo.
    echo skyfind gui exited with code %RC%.
    REM keep the window readable when launched by double-click
    if "%cmdcmdline:~0,7%"=="cmd /c " pause
)
exit /b %RC%
