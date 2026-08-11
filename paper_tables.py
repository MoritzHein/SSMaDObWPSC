# LaTeX tables of the two case studies added for the paper.
#
# Usage: paper_tables.py [--scaling=settings/scaling_campaign.json]
#                        [--central=settings/centralized_campaign.json]
#
# Prints the bodies of
#
#   tab:scaling  - solution time against the depth of the architecture
#   tab:central  - decomposed against centralized solution on small architectures
#
# so that the numbers in the paper are generated from results/run_<N>/ rather
# than copied by hand. Two conventions matter and are easy to get wrong:
#
#   * the decomposed solution time is t_c = sum_l max_r t_{c,r,l}, summed over
#     the workflows, taken from time.json["time_list"] (indexed
#     [workflow][level][region]). time.json["overall_time"] is the wall clock of
#     the whole placement phase including architecture generation and file I/O,
#     which is five to a hundred times larger and is NOT what the base paper
#     reports;
#   * a solve that stopped on its time limit returns an incumbent. Its achieved
#     gap is in time.json["own_mipgap"] for the decomposed subproblems and in
#     time.json["comparison_mipgap"] for the centralized run. The tables mark
#     such an entry instead of presenting it as an optimum.
import argparse
import json
import math
import os

GAP_TOL = 1e-4          # above this a solve is reported as an incumbent


def flat(x):
    if isinstance(x, list):
        for y in x:
            yield from flat(y)
    else:
        yield x


def load(num_run, name):
    path = os.path.join("results", f"run_{num_run}", name)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def level_max_time(time_list):
    """t_c = sum over levels of the slowest region, summed over the workflows."""
    total = 0.0
    for workflow in time_list:
        for level in workflow:
            vals = [v for v in flat(level) if v is not None]
            if vals:
                total += max(vals)
    return total


def total_time(time_list):
    return sum(v for v in flat(time_list) if v is not None)


def mean(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return sum(v) / len(v) if v else float("nan")


def paired(det_rows, stoch_rows, key="overall_cost"):
    d = {r["simulation_seed"]: r for r in det_rows}
    s = {r["simulation_seed"]: r for r in stoch_rows}
    v = [s[k][key] - d[k][key] for k in sorted(set(d) & set(s))]
    if not v:
        return float("nan"), 0.0, 0, 0
    m = sum(v) / len(v)
    if len(v) < 2:
        return m, 0.0, len(v), sum(1 for x in v if x < 0)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
    return m, 1.96 * sd / math.sqrt(len(v)), len(v), sum(1 for x in v if x < 0)


def worst_gap(time_json, key):
    gaps = [v for v in flat(time_json.get(key, [])) if v is not None]
    return max(gaps) if gaps else 0.0


def architecture(num_run):
    """Physical devices and subproblems per workflow of a run's architecture."""
    path = os.path.join("cloud", f"run_{num_run}", "general_config.json")
    with open(path, "r") as f:
        cfg = json.load(f)
    devices = sum(cfg["num_nodes_in_level"][-1])
    subproblems = sum(len(level) for level in cfg["num_nodes_in_level"])
    return devices, subproblems


def scaling_table(manifest_path):
    with open(manifest_path, "r") as f:
        man = json.load(f)
    by_cell = {(v["num_levels"], v["policy"]): int(k) for k, v in man["runs"].items()}
    lines = []
    for levels in man["levels"]:
        cells = {}
        for pol in ("det", "stoch"):
            num_run = by_cell.get((levels, pol))
            t = load(num_run, "time.json") if num_run else None
            rows = load(num_run, "results_seeds.json") if num_run else None
            cells[pol] = (num_run, t, rows)
        det_run, det_t, det_rows = cells["det"]
        _, stoch_t, stoch_rows = cells["stoch"]
        if det_t is None or stoch_t is None:
            lines.append(f"% levels {levels}: incomplete")
            continue
        devices, subproblems = architecture(det_run)
        capped = max(worst_gap(det_t, "own_mipgap"), worst_gap(stoch_t, "own_mipgap"))
        mark = "$^\\ast$" if capped > GAP_TOL else ""
        if det_rows and stoch_rows:
            cd, cs = mean(det_rows, "overall_cost"), mean(stoch_rows, "overall_cost")
            diff, _, _, _ = paired(det_rows, stoch_rows)
            cost = f"${cd:.0f}$&${cs:.0f}$&${100 * diff / cd:+.1f}$"
        else:
            cost = "\\multicolumn{3}{c}{not simulated}"
        lines.append(
            f"${levels}$&${devices}$&${subproblems}$&"
            f"${level_max_time(det_t['time_list']):.2f}$&"
            f"${level_max_time(stoch_t['time_list']):.2f}$&{cost}{mark}\\\\")
        # The total effort of one machine is quoted in the text, not in the
        # table, which has no room for two more columns.
        lines.append(f"% levels {levels}: total solution time "
                     f"{total_time(det_t['time_list']):.1f} s deterministic, "
                     f"{total_time(stoch_t['time_list']):.1f} s stochastic")
    return lines


def centralized_table(manifest_path):
    with open(manifest_path, "r") as f:
        man = json.load(f)
    by_cell = {(v["physical_nodes"], v["ram"], v["policy"]): int(k)
               for k, v in man["runs"].items()}
    label = {"off": "aggr.", "l2": "elig., $\\ell_2$"}
    lines = []
    for nodes in sorted({v["physical_nodes"] for v in man["runs"].values()}):
        first = True
        for ram in ("off", "l2"):
            for pol in ("det", "stoch"):
                num_run = by_cell.get((nodes, ram, pol))
                rows = load(num_run, "results_seeds.json") if num_run else None
                t = load(num_run, "time.json") if num_run else None
                if not rows or t is None:
                    lines.append(f"% {nodes} devices {ram} {pol}: pending")
                    continue
                dec = mean(rows, "overall_cost")
                cen = mean(rows, "overall_cost_comparison")
                td = level_max_time(t["time_list"])
                tc = sum(v for v in flat(t["comparison_time"]) if v is not None)
                mark = "$^\\ast$" if worst_gap(t, "comparison_mipgap") > GAP_TOL else ""
                head = f"${nodes}$" if first else ""
                first = False
                lines.append(
                    f"{head}&{label[ram]}&{'det.' if pol == 'det' else 'stoch.'}&"
                    f"${dec:.0f}$&${cen:.0f}$&${100 * (dec - cen) / dec:+.1f}$&"
                    f"${td:.1f}$&${tc:.0f}${mark}\\\\")
        lines.append("\\hline")
    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling", default=os.path.join("settings", "scaling_campaign.json"))
    parser.add_argument("--central", default=os.path.join("settings", "centralized_campaign.json"))
    args = parser.parse_args()

    print("% ---- tab:scaling ----")
    for line in scaling_table(args.scaling):
        print(line)
    print()
    print("% ---- tab:central ----")
    for line in centralized_table(args.central):
        print(line)
