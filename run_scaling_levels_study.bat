:: Scalability study over the number of hierarchical levels.
::
:: Runs placement + selection simulation for runs 330..335, i.e. two, three and
:: four levels (40 / 400 / 4000 physical devices) under both selection policies,
:: at w_3 = 1 with the eligibility + l2 capacity model.
::
:: The four-level runs (334, 335) are simulated as well, so comparison = True is
:: needed there: selection_simulation.py reads requirements_cloud_full.json for
:: the request rates and the per-node RAM limits, and placement.py writes that
:: flattened architecture only when comparison is on. That also puts the
:: centralized solve over 4000 devices into the placement phase - its latency
:: matrix has 16 million entries, so expect the two runs to be the slow ones.
::
:: Generate the parameter files first:  scaling_levels_experiments.py
:: Tabulate the results afterwards:     paper_tables.py
::
:: Expect hours rather than minutes. The three-level simulation alone takes
:: about five and a half minutes per request stream, so the ten streams of runs
:: 332 and 333 dominate the total. Split them across cores with --runs if
:: needed.
@echo off
setlocal
set PY=.optifaasvenv\Scripts\python.exe

for /l %%x in (330, 1, 335) do (
    echo === run %%x ===
    %PY% placement.py --num_run=%%x
    %PY% selection_simulation.py --num_run=%%x
)

%PY% resimulate_seeds.py --manifest=settings\scaling_campaign.json --seeds=10

echo.
echo Study complete. Tabulate with:
echo     %PY% paper_tables.py
endlocal
