@echo off
REM ===========================================================================
REM  ClipFinder launcher
REM  Double-click THIS file to start the app. It:
REM    1. opens ClipFinder in your default browser (after a short pause so the
REM       server has time to start), and
REM    2. starts the local server in this window.
REM  Keep this .bat in the same folder as clipfinder.py.
REM  IMPORTANT: leave this window OPEN while you use the app. It IS the server.
REM  Closing this window stops the app.
REM ===========================================================================

REM --- Open the browser to the app after a 3-second delay, without blocking. ---
REM  "start" launches it in parallel; the timeout gives the server time to boot.
start "" cmd /c "timeout /t 3 >nul & start http://localhost:8000"

REM --- Start the server (this keeps running in THIS window). ---
python "%~dp0clipfinder.py"

REM If the window closes instantly, an error happened, run from a terminal to see it:
REM   open Command Prompt in this folder and type:  python clipfinder.py
pause
