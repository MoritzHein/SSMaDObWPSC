:: Scaling study against the CENTRALIZED solution (Sec. IV-C, Table 3).
:: Runs placement + selection simulation for runs 126..129 (parameter files
:: with increasing node counts; create them analogously to
:: scaling_experiments.py).
:: NOTE: the absolute paths point to the original author's machine - adapt
:: the Python interpreter and script paths to your own checkout/venv.
:: Visualize the results afterwards with: visualization_time.py --num_run=126
for /l %%x in (126, 1,129) do (
    C:\Users\smjsadam\Documents\2025-scheduling-workflows-faas\venv\Scripts\python.exe C:\Users\smjsadam\Documents\2025-scheduling-workflows-faas\placement.py --num_run=%%x
    C:\Users\smjsadam\Documents\2025-scheduling-workflows-faas\venv\Scripts\python.exe C:\Users\smjsadam\Documents\2025-scheduling-workflows-faas\selection_simulation.py --num_run=%%x
)
