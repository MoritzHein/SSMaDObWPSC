# Visualization of the sensitivity study.
#
# Usage: visualization_sensitivity.py --num_run=<N> [--params=<p1,p2,...>]
#        (figures are saved to results/run_<N>/)
#
# --params (alias --param) restricts the per-parameter figures (tornado,
# response curves, adaptation offset, regression bars) to a comma-separated
# subset of request_rate,time,speedup,latency,ram. Subset figures are saved
# with the selected parameters as a filename suffix so the full-study figures
# are not overwritten; the regression is still FITTED on all factor columns
# (only the displayed bars are filtered), so regression_summary.json is
# unaffected by the selection.
#
# --comparison=False drops the comparison solution from every figure: the
# two-column figures collapse to a single decomposed panel (and the
# distribution/regression show only the decomposed series). Figures are then
# saved with a "_decomposed" filename suffix so the full figures are kept.
# Default: --comparison=True (unchanged behavior).
#
# Produces, when the corresponding data is present:
#   - sensitivity_tornado          : per-parameter overall_cost range from the
#                                    OFAT scale sweep (run_scale_sweep.py),
#                                    ranking the effect of each individual
#                                    uncertainty.
#   - sensitivity_response_curves  : overall_cost vs. scale factor, one line per
#                                    parameter (slope = sensitivity); with
#                                    adapted-placement data (run_scale_sweep.py
#                                    --adapt) the re-planned costs are overlaid
#                                    as dashed lines.
#   - sensitivity_adaptation_offset: offset of the fixed nominal placement to
#                                    the adapted re-planning baseline (absolute
#                                    and relative), i.e. the regret of not
#                                    re-planning under the perturbed parameters.
#   - sensitivity_distribution     : spread of overall_cost across the random
#                                    combined study (run_sensitivity.py) vs. the
#                                    baseline.
#   - sensitivity_regression       : standardized regression coefficients of
#                                    overall_cost on the per-parameter factors of
#                                    the combined study (a statistical estimate of
#                                    the individual effects); also written as
#                                    regression_summary.json.
# Each figure covers both the decomposed solution (overall_cost) and the
# comparison solution (overall_cost_comparison); whether the comparison is the
# centralized optimization or the all-on-cloud heuristic is read from
# settings/parameters_<N>.json and stated in the titles/legends.
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# Reuse the paper plotting style when available, but degrade gracefully so the
# script runs without a LaTeX toolchain (mathtext labels render either way).
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:
    pass

# Human-readable axis labels for each parameter key.
PARAM_LABELS = {
    "request_rate": r"demand $\lambda$",
    "time": "runtime",
    "speedup": "speedup",
    "latency": r"latency $L$",
    "ram": "RAM",
}
# Fixed per-parameter (color, marker) assignment: identity stays consistent
# across all figures and is never encoded by color alone (the marker is the
# color-vision-deficiency fallback).
PARAM_STYLE = {
    "request_rate": ("C0", "o"),
    "time": ("C1", "s"),
    "speedup": ("C2", "^"),
    "latency": ("C3", "D"),
    "ram": ("C4", "v"),
}


def make_targets(comp_label):
    """The two cost columns of every figure: the decomposed solution and the
    comparison solution, the latter named by what it actually is
    ("centralized" or "heuristic", from the run's settings)."""
    return [("overall_cost", "decomposed"), ("overall_cost_comparison", comp_label)]


def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def load_csv(path):
    try:
        with open(path, "r", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return None


def label(p):
    return PARAM_LABELS.get(p, p)


def style(p):
    return PARAM_STYLE.get(p, ("C7", "x"))


def save(fig, export_path, name, suffix=""):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(export_path, f"{name}{suffix}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}{suffix}.pdf / .png")


def _finite(values):
    return [v for v in values if isinstance(v, (int, float)) and np.isfinite(v)]


# ---------------------------------------------------------------- tornado
def plot_tornado(sweep, export_path, targets, suffix=""):
    params = list(sweep.get("params", {}).keys())
    if not params:
        return
    fig, axes = plt.subplots(1, len(targets),
                             figsize=(4.5 * len(targets), 0.6 * len(params) + 2))
    for ax, (target, title) in zip(np.atleast_1d(axes), targets):
        base = sweep.get("baseline", {}).get(target)
        rows = []
        for p in params:
            costs = _finite(sweep["params"][p].get(target, []))
            if not costs:
                continue
            lo, hi = min(costs), max(costs)
            rows.append((p, lo, hi, hi - lo))
        if not rows:
            ax.set_visible(False)
            continue
        rows.sort(key=lambda r: r[3])  # widest impact on top (barh draws bottom-up)
        if base is None:
            base = np.mean([(lo + hi) / 2 for _, lo, hi, _ in rows])
        ypos = np.arange(len(rows))
        ax.barh(ypos, [hi - lo for _, lo, hi, _ in rows],
                left=[lo for _, lo, _, _ in rows], color="#4878CF", alpha=0.85)
        ax.axvline(base, color="k", linestyle="--", linewidth=1, label="baseline")
        ax.set_yticks(ypos)
        ax.set_yticklabels([label(p) for p, *_ in rows])
        ax.set_xlabel(r"Overall cost $C$")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Effect of individual uncertainties (scale sweep)")
    save(fig, export_path, "sensitivity_tornado", suffix)


# -------------------------------------------------------- response curves
def plot_response_curves(sweep, export_path, targets, suffix=""):
    params = list(sweep.get("params", {}).keys())
    if not params:
        return
    any_adapted = False
    fig, axes = plt.subplots(1, len(targets), figsize=(4.5 * len(targets), 3.5),
                             sharex=True, sharey=True)
    for ax, (target, title) in zip(np.atleast_1d(axes), targets):
        base = sweep.get("baseline", {}).get(target)
        for p in params:
            f = sweep["params"][p].get("factors", [])
            c = sweep["params"][p].get(target, [])
            col, mark = style(p)
            pts = [(x, y) for x, y in zip(f, c)
                   if isinstance(y, (int, float)) and np.isfinite(y)]
            if pts:
                xs, ys = zip(*sorted(pts))
                ax.plot(xs, ys, marker=mark, color=col, label=label(p))
            # Adapted re-planning baseline: same color/marker as the parameter,
            # dashed with open markers (linestyle carries nominal vs. adapted).
            a = sweep["params"][p].get(f"adapted_{target}", [])
            apts = [(x, y) for x, y in zip(f, a)
                    if isinstance(y, (int, float)) and np.isfinite(y)]
            if apts:
                any_adapted = True
                xs, ys = zip(*sorted(apts))
                ax.plot(xs, ys, marker=mark, color=col, linestyle="--",
                        markerfacecolor="none")
        if base is not None:
            ax.axhline(base, color="k", linestyle=":", linewidth=1)
        ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("scale factor")
        ax.set_ylabel(r"Overall cost $C$")
        ax.set_title(title)
    handles, names = np.atleast_1d(axes)[0].get_legend_handles_labels()
    if any_adapted:
        handles = handles + [Line2D([0], [0], color="k", linestyle="-"),
                             Line2D([0], [0], color="k", linestyle="--")]
        names = names + ["nominal placement", "adapted placement"]
    np.atleast_1d(axes)[0].legend(handles, names, loc="best", fontsize=8)
    fig.suptitle("Cost response to each uncertainty")
    save(fig, export_path, "sensitivity_response_curves", suffix)


# ------------------------------------------------------- adaptation offset
def plot_adaptation_offset(sweep, export_path, targets, suffix=""):
    """Offset of the fixed nominal placement to the adapted re-planning
    baseline (run_scale_sweep.py --adapt), per parameter and scale factor.
    Top row: absolute offset, bottom row: relative offset. Positive = keeping
    the nominal placement is more expensive than re-planning (regret)."""
    params = list(sweep.get("params", {}).keys())
    if not params:
        return
    have = [(target, title) for target, title in targets
            if any(_finite(sweep["params"][p].get(f"adapted_{target}", []))
                   for p in params)]
    if not have:
        print("  no adapted data in scale_sweep_summary.json "
              "(run run_scale_sweep.py --adapt) -> skipping adaptation offset")
        return
    ncols = len(have)
    fig, axes = plt.subplots(2, ncols, figsize=(4.5 * ncols, 6),
                             sharex=True, sharey='row', squeeze=False)
    for col_i, (target, title) in enumerate(have):
        ax_abs, ax_rel = axes[0][col_i], axes[1][col_i]
        for p in params:
            f = sweep["params"][p].get("factors", [])
            nom = sweep["params"][p].get(target, [])
            ada = sweep["params"][p].get(f"adapted_{target}", [])
            pts = [(x, n, a) for x, n, a in zip(f, nom, ada)
                   if isinstance(n, (int, float)) and np.isfinite(n)
                   and isinstance(a, (int, float)) and np.isfinite(a) and a != 0]
            if not pts:
                continue
            pts.sort()
            xs = [x for x, _, _ in pts]
            col, mark = style(p)
            ax_abs.plot(xs, [n - a for _, n, a in pts],
                        marker=mark, color=col, label=label(p))
            ax_rel.plot(xs, [(n - a) / a * 100.0 for _, n, a in pts],
                        marker=mark, color=col)
        for ax in (ax_abs, ax_rel):
            ax.axhline(0, color="k", linewidth=0.8)
            ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8)
        ax_abs.set_title(title)
        ax_rel.set_xlabel("scale factor")
    axes[0][0].set_ylabel(r"$C_\mathrm{nominal} - C_\mathrm{adapted}$")
    axes[1][0].set_ylabel(r"$(C_\mathrm{nom} - C_\mathrm{ad})\,/\,C_\mathrm{ad}$ [%]")
    axes[0][0].legend(loc="best", fontsize=8)
    fig.suptitle("Regret of keeping the nominal placement vs. re-planning")
    save(fig, export_path, "sensitivity_adaptation_offset", suffix)


# ----------------------------------------------------------- distribution
def plot_distribution(summary, export_path, targets, suffix=""):
    scenarios = summary.get("scenarios", [])
    name_dec = targets[0][1]
    # With --comparison=False the targets carry only the decomposed solution;
    # drop the comparison series entirely.
    name_comp = targets[1][1] if len(targets) > 1 else None
    oc = _finite([r.get("overall_cost") for r in scenarios])
    occ = (_finite([r.get("overall_cost_comparison") for r in scenarios])
           if name_comp else [])
    if not oc and not occ:
        return
    base = summary.get("baseline", {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    data, labels, bases = [], [], []
    if oc:
        data.append(oc); labels.append(name_dec); bases.append(base.get("overall_cost"))
    if occ:
        data.append(occ); labels.append(name_comp); bases.append(base.get("overall_cost_comparison"))
    parts = ax1.violinplot(data, showmeans=False, showextrema=True)
    ax1.boxplot(data, widths=0.15)
    ax1.set_xticks(range(1, len(labels) + 1))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel(r"Overall cost $C$")
    for i, b in enumerate(bases, start=1):
        if b is not None:
            ax1.hlines(b, i - 0.4, i + 0.4, color="r", linestyle="--", linewidth=1)
    ax1.set_title("Combined-study spread")

    if oc:
        ax2.hist(oc, bins=min(20, max(5, len(oc) // 2)), alpha=0.6, label=name_dec)
    if occ:
        ax2.hist(occ, bins=min(20, max(5, len(occ) // 2)), alpha=0.6, label=name_comp)
    if base.get("overall_cost") is not None:
        ax2.axvline(base["overall_cost"], color="C0", linestyle="--", linewidth=1)
    if name_comp and base.get("overall_cost_comparison") is not None:
        ax2.axvline(base["overall_cost_comparison"], color="C1", linestyle="--", linewidth=1)
    ax2.set_xlabel(r"Overall cost $C$")
    ax2.set_ylabel("count")
    ax2.legend(fontsize=8)
    ax2.set_title("Distribution")
    fig.suptitle("Sensitivity distribution (combined study)")
    save(fig, export_path, "sensitivity_distribution", suffix)


# ------------------------------------------------------------- regression
def _standardized_regression(rows, target, param_cols):
    """Standardized regression coefficients of `target` on the per-parameter
    factor columns. Returns (coef_dict, r2, dropped_constant_cols)."""
    X, y = [], []
    for r in rows:
        try:
            yv = float(r[target])
        except (KeyError, ValueError, TypeError):
            continue
        xv = []
        ok = True
        for c in param_cols:
            try:
                xv.append(float(r[c]))
            except (KeyError, ValueError, TypeError):
                ok = False
                break
        if ok and np.isfinite(yv):
            X.append(xv)
            y.append(yv)
    if len(y) < 2:
        return {}, None, []
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    col_std = X.std(axis=0)
    keep = col_std > 1e-12
    dropped = [c for c, k in zip(param_cols, keep) if not k]
    kept_cols = [c for c, k in zip(param_cols, keep) if k]
    if not kept_cols or y.std() < 1e-12:
        return {}, None, dropped
    Xs = (X[:, keep] - X[:, keep].mean(axis=0)) / col_std[keep]
    ys = (y - y.mean()) / y.std()
    beta, _, _, _ = np.linalg.lstsq(Xs, ys, rcond=None)
    pred = Xs @ beta
    r2 = float(1.0 - np.sum((ys - pred) ** 2) / np.sum(ys ** 2))
    coef = {c.replace("_factor_mean", ""): float(b) for c, b in zip(kept_cols, beta)}
    return coef, r2, dropped


def plot_regression(csv_rows, export_path, targets, selected=None, suffix=""):
    if not csv_rows:
        return
    param_cols = [c for c in csv_rows[0].keys() if c.endswith("_factor_mean")]
    if not param_cols:
        print("  regression: no *_factor_mean columns in summary.csv; skipped")
        return
    out = {}
    for target, name in targets:
        coef, r2, dropped = _standardized_regression(csv_rows, target, param_cols)
        out[name] = {"coefficients": coef, "r2": r2, "dropped_constant": dropped}

    params = sorted({p for v in out.values() for p in v["coefficients"]})
    # --params only filters the DISPLAYED bars; the fit above always uses all
    # factor columns so the coefficients (and regression_summary.json) do not
    # change with the selection.
    if selected is not None:
        params = [p for p in params if p in selected]
    if not params:
        if selected is not None and any(v["coefficients"] for v in out.values()):
            print("  regression: none of the selected --params has a fitted "
                  "coefficient; skipped figure")
        else:
            print("  regression: all factor columns constant (use per_element=false "
                  "in the combined study for a meaningful estimate); skipped figure")
        with open(os.path.join(export_path, f"regression_summary{suffix}.json"), "w") as f:
            json.dump(out, f, indent=4)
        return

    fig, ax = plt.subplots(figsize=(7, 3.5))
    width = 0.38
    xpos = np.arange(len(params))
    for i, (target, name) in enumerate(targets):
        coef = out[name]["coefficients"]
        vals = [coef.get(p, 0.0) for p in params]
        r2 = out[name]["r2"]
        lbl = f"{name}" + (f" ($R^2$={r2:.2f})" if r2 is not None else "")
        # Center the group of bars for any number of targets (2 -> +-width/2,
        # 1 -> centered on the tick).
        ax.bar(xpos + (i - (len(targets) - 1) / 2) * width, vals, width, label=lbl)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([label(p) for p in params])
    ax.set_ylabel("standardized regression coef.")
    ax.set_title("Estimated individual effects (combined study regression)")
    ax.legend(fontsize=8)
    save(fig, export_path, "sensitivity_regression", suffix)

    with open(os.path.join(export_path, f"regression_summary{suffix}.json"), "w") as f:
        json.dump(out, f, indent=4)
    print(f"  saved regression_summary{suffix}.json")


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected True/False, got {v!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_run", required=False, type=int, default=None)
    parser.add_argument("--params", "--param", dest="params", required=False,
                        type=str, default=None,
                        help="Comma-separated subset of parameters to plot "
                             f"(default: all). Choose from {','.join(PARAM_LABELS)}. "
                             "Subset figures get the selection as a filename "
                             "suffix so the full figures are kept.")
    parser.add_argument("--comparison", required=False, type=_str2bool,
                        default=True,
                        help="Also plot the comparison solution (default: True). "
                             "--comparison=False collapses every figure to the "
                             "decomposed solution only; those figures get a "
                             "'_decomposed' filename suffix.")
    args = parser.parse_args()
    num_run = args.num_run
    if num_run is None:
        while True:
            try:
                num_run = int(input("Please enter num_run: "))
                break
            except ValueError:
                print("num_run must be an integer. Please try again.")

    export_path = os.path.join("results", f"run_{num_run}")
    if not os.path.exists(export_path):
        raise SystemExit(f"{export_path} does not exist; run the study first.")
    sens_dir = os.path.join(export_path, "sensitivity")

    # Name the comparison by what it actually is, so every figure states
    # directly whether it shows the heuristic or the centralized solution.
    settings = load_json(os.path.join("settings", f"parameters_{num_run}.json"))
    if settings is None:
        comp_label = "comparison"
    else:
        comp_label = "heuristic" if settings.get("heuristic") else "centralized"
    targets = make_targets(comp_label)
    if not args.comparison:
        targets = targets[:1]

    # Optional parameter selection: restrict the per-parameter figures to a
    # subset. Validated against the known parameter keys; subset figures are
    # saved with the selection as a filename suffix so a later full run does
    # not have to regenerate anything.
    selected = None
    suffix = ""
    if args.params:
        selected = [p.strip() for p in args.params.split(",") if p.strip()]
        unknown = [p for p in selected if p not in PARAM_LABELS]
        if unknown:
            raise SystemExit(f"Unknown parameter(s) {unknown}. "
                             f"Choose from {list(PARAM_LABELS)}.")
        if set(selected) != set(PARAM_LABELS):
            suffix = "_" + "_".join(selected)
    if not args.comparison:
        suffix += "_decomposed"

    sweep = load_json(os.path.join(sens_dir, "scale_sweep_summary.json"))
    summary = load_json(os.path.join(sens_dir, "summary.json"))
    csv_rows = load_csv(os.path.join(sens_dir, "summary.csv"))

    if sweep and selected is not None:
        missing = [p for p in selected if p not in sweep.get("params", {})]
        if missing:
            print(f"  note: {missing} not in the scale-sweep data; plotting the rest")
        sweep["params"] = {p: v for p, v in sweep.get("params", {}).items()
                           if p in selected}

    print(f"Saving figures to {export_path} "
          f"(comparison = {comp_label if args.comparison else 'off'}"
          + (f", params = {','.join(selected)}" if selected is not None else "") + ")")
    if sweep:
        plot_tornado(sweep, export_path, targets, suffix)
        plot_response_curves(sweep, export_path, targets, suffix)
        plot_adaptation_offset(sweep, export_path, targets, suffix)
    else:
        print("  no scale_sweep_summary.json (run run_scale_sweep.py) -> "
              "skipping tornado + response curves + adaptation offset")
    if summary:
        plot_distribution(summary, export_path, targets,
                          "" if args.comparison else "_decomposed")
    else:
        print("  no summary.json (run run_sensitivity.py) -> skipping distribution")
    if csv_rows:
        plot_regression(csv_rows, export_path, targets, selected, suffix)
    else:
        print("  no summary.csv -> skipping regression")


if __name__ == "__main__":
    main()
