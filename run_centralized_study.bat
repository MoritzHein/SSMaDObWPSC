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
:: minutes at sixteen, and open-ended at twenty - comparison_time_limit is None,
:: so every solve runs to proven optimality instead of returning an incumbent
:: after 1800 s. Budget for the twenty-device runs (352..355) taking many hours.
:: paper_tables.py still marks any run whose comparison_mipgap exceeds 1e-4, so
:: a capped solve would never be presented as an optimum.
@echo off
setlocal
set PY=.optifaasvenv\Scripts\python.exe

for /l %%x in (340, 1, 355) do (
    echo === run %%x ===
    %PY% placement.py --num_run=%%x
    %PY% selection_simulation.py --num_run=%%x
)

%PY% resimulate_seeds.py --manifest=settings\centralized_campaign.json --seeds=10

echo.
echo Study complete. Tabulate with:
echo     %PY% paper_tables.py
endlocal
