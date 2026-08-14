:: Comparison of the decomposed placement against the centralized solve.
::
:: Runs placement + selection simulation for runs 340..355, i.e. 8 / 12 / 16 / 20
:: physical devices crossed with the two capacity models and the two selection
:: policies, at w_3 = 1. heuristic = False makes the comparison run the
:: centralized optimization on the flattened architecture rather than the
:: all-on-cloud heuristic.
::
:: Generate the parameter files first:  centralized_experiments.py
:: Tabulate the results afterwards:     paper_tables.py
::
:: The centralized problem is a nonconvex program over all physical devices and
:: is what makes this study slow: a few seconds at eight devices, several
:: minutes at sixteen, and open-ended at twenty. Whether a solve may stop early
:: is decided by comparison_time_limit in the parameter files, i.e. by
:: COMPARISON_TIME_LIMIT in centralized_experiments.py, not here.
:: paper_tables.py marks any run whose comparison_mipgap exceeds 1e-4, so a
:: capped solve is never presented as an optimum.
::
:: ---------------------------------------------------------------------------
:: PARALLELISM
::
:: Phase 1, placement - SEQUENTIAL, and here that is not optional. Table
:: tab:central of the paper reports both t_dec and t_cen from
:: results/run_<N>/time.json, and the centralized solve is the quantity the
:: table is about. Concurrent Gurobi solves would share the cores and the times
:: would stop being solution times.
::
:: Phase 2, simulation - PARALLEL, one worker per run, each running its
:: baseline stream and then its request streams. These architectures are small,
:: so all sixteen runs fit comfortably side by side.
::
:: MAXJOBS caps concurrent workers (default 8). Per-run logs go to
:: logs\centralized\run_<N>.log.
:: ---------------------------------------------------------------------------
@echo off
setlocal enabledelayedexpansion

set PY=.optifaasvenv\Scripts\python.exe
set PINGEXE=%SystemRoot%\System32\ping.exe
set SEEDS=10
set LOGDIR=logs\centralized
set MANIFEST=settings\centralized_campaign.json
set FIRST=340
set LAST=355
set SIMRUNS=340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355
if "%MAXJOBS%"=="" set MAXJOBS=8

if /i "%~1"=="--worker" goto worker

if not exist "%LOGDIR%" mkdir "%LOGDIR%"
del /q "%LOGDIR%\*.done" 2>nul
del /q "%LOGDIR%\*.started" 2>nul
del /q "%LOGDIR%\*.log" 2>nul

echo ============================================================
echo Phase 1/2: placement, runs %FIRST%..%LAST% (sequential, centralized solves are timed)
echo ============================================================
for /l %%x in (%FIRST%, 1, %LAST%) do (
    echo === placement run %%x ===
    %PY% placement.py --num_run=%%x
    if errorlevel 1 echo *** placement run %%x FAILED
)

echo.
echo ============================================================
echo Phase 2/2: simulation on %SEEDS% request streams, up to %MAXJOBS% runs at a time
echo ============================================================
for %%x in (%SIMRUNS%) do (
    call :throttle
    echo === starting simulation worker for run %%x ===
    type nul > "%LOGDIR%\run_%%x.started"
    start "sim %%x" /B cmd /c ""%~f0" --worker %%x"
)
call :waitall
call :report

echo.
echo Study complete. Tabulate with:
echo     %PY% paper_tables.py
endlocal
goto :eof

:: ---------------------------------------------------------------------------
:worker
set RUN=%~2
%PY% selection_simulation.py --num_run=%RUN% > "%LOGDIR%\run_%RUN%.log" 2>&1
if errorlevel 1 (
    echo *** baseline simulation failed >> "%LOGDIR%\run_%RUN%.log"
    >"%LOGDIR%\run_%RUN%.done" echo EXIT=1
    goto :eof
)
%PY% resimulate_seeds.py --manifest=%MANIFEST% --seeds=%SEEDS% --runs=%RUN% >> "%LOGDIR%\run_%RUN%.log" 2>&1
>"%LOGDIR%\run_%RUN%.done" echo EXIT=%errorlevel%
goto :eof

:: Sleep %1 seconds. timeout.exe refuses to run with redirected stdin, which is
:: what happens when this file is started from a non-interactive shell.
:sleep
set /a _pings=%~1+1
%PINGEXE% -n !_pings! 127.0.0.1 >nul 2>&1
goto :eof

:throttle
set /a running=0
for %%r in (%SIMRUNS%) do (
    if exist "%LOGDIR%\run_%%r.started" if not exist "%LOGDIR%\run_%%r.done" set /a running+=1
)
if !running! geq %MAXJOBS% (
    call :sleep 5
    goto throttle
)
goto :eof

:waitall
set /a left=0
for %%r in (%SIMRUNS%) do (
    if not exist "%LOGDIR%\run_%%r.done" set /a left+=1
)
if !left! gtr 0 (
    call :sleep 15
    goto waitall
)
goto :eof

:report
echo.
echo --- workers that did not exit cleanly ---
set /a bad=0
for %%r in (%SIMRUNS%) do (
    set /p code=<"%LOGDIR%\run_%%r.done"
    if not "!code!"=="EXIT=0" (
        echo   run %%r: !code!  see %LOGDIR%\run_%%r.log
        set /a bad+=1
    )
)
if !bad!==0 echo   none - all %SEEDS%-stream simulations completed
goto :eof
