:: Scaling study against the CENTRALIZED solution (Sec. IV-C, Table 3).
:: Runs placement + selection simulation for runs 126..129 (parameter files
:: with increasing node counts; create them analogously to
:: scaling_experiments.py).
:: NOTE: the absolute paths point to the original author's machine - adapt
:: the Python interpreter and script paths to your own checkout/venv.
:: Visualize the results afterwards with: visualization_time.py --num_run=126
for /l %%x in (126, 1,129) do (
    C:\Users\smmzhein\AppData\Local\anaconda3\envs\OptiFaas\python.exe Z:\Documents\OptiFaas\IEA_LSWPISCUINP\placement.py --num_run=%%x
    C:\Users\smmzhein\AppData\Local\anaconda3\envs\OptiFaas\python.exe Z:\Documents\OptiFaas\IEA_LSWPISCUINP\selection_simulation.py --num_run=%%x
)
