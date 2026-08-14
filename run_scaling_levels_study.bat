:: Scalability study over the number of hierarchical levels.
::
:: Runs placement + selection simulation for runs 330..335, i.e. two, three and
:: four levels (40 / 400 / 4000 physical devices) under both selection policies,
:: at w_3 = 1 with the eligibility + l2 capacity model.
::
:: comparison = True is needed even where the centralized solution is not of
:: interest: selection_simulation.py reads requirements_cloud_full.json for the
:: request rates and the per-node RAM limits, and placement.py writes that
:: flattened architecture only when comparison is on. At four levels its latency
:: matrix has 16 million entries, so building it is what makes runs 334 and 335
:: the slow ones of the placement phase.
::
:: Generate the parameter files first:  scaling_levels_experiments.py
:: Tabulate the results afterwards:     paper_tables.py
::
:: ---------------------------------------------------------------------------
:: PARALLELISM
::
:: Phase 1, placement - SEQUENTIAL, and here that is not optional. Table
:: tab:scaling of the paper reports t_c = sum_l max_r t_{c,r,l} from
:: results/run_<N>/time.json, so the subproblems must not compete for cores
:: while they are being timed.
::
:: Phase 2, simulation - PARALLEL, one worker per run, each running its
:: baseline stream and then its request streams. Simulation is not timed by any
:: table, and every run touches only its own files.
::
:: SIMRUNS selects which runs are simulated at all. The default excludes the
:: four-level runs 334 and 335: the paper reports solution times only for that
:: depth ("not simulated" in tab:scaling), because one four-level stream costs
:: far more than the three-level ones while the arrival rate, and with it the
:: number of requests, is unchanged. Override to include them, e.g.
::     set SIMRUNS=330 331 332 333 334 335 & run_scaling_levels_study.bat
:: MAXJOBS caps concurrent workers (default 4; there are at most six runs).
:: Per-run logs go to logs\scaling\run_<N>.log.
:: ---------------------------------------------------------------------------
@echo off
setlocal enabledelayedexpansion

set PY=.optifaasvenv\Scripts\python.exe
set PINGEXE=%SystemRoot%\System32\ping.exe
set SEEDS=10
set LOGDIR=logs\scaling
set MANIFEST=settings\scaling_campaign.json
set FIRST=330
set LAST=335
if "%SIMRUNS%"=="" set SIMRUNS=330 331 332 333
if "%MAXJOBS%"=="" set MAXJOBS=4

if /i "%~1"=="--worker" goto worker

if not exist "%LOGDIR%" mkdir "%LOGDIR%"
del /q "%LOGDIR%\*.done" 2>nul
del /q "%LOGDIR%\*.started" 2>nul
del /q "%LOGDIR%\*.log" 2>nul

echo ============================================================
echo Phase 1/2: placement, runs %FIRST%..%LAST% (sequential, solves are timed)
echo ============================================================
for /l %%x in (%FIRST%, 1, %LAST%) do (
    echo === placement run %%x ===
    %PY% placement.py --num_run=%%x
    if errorlevel 1 echo *** placement run %%x FAILED
)

echo.
echo ============================================================
echo Phase 2/2: simulation of runs %SIMRUNS% on %SEEDS% request streams
echo            (parallel, up to %MAXJOBS% at a time)
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
