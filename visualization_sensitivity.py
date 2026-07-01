# Visualization of the sensitivity study.
#
# Usage: visualization_sensitivity.py --num_run=<N>
#        (figures are saved to results/run_<N>/)
#
# Produces, when the corresponding data is present:
#   - sensitivity_tornado        : per-parameter overall_cost range from the
#                                  OFAT scale sweep (run_scale_sweep.py),
#                                  ranking the effect of each individual
#                                  uncertainty.
#   - sensitivity_response_curves: overall_cost vs. scale factor, one line per
#                                  parameter (slope = sensitivity).
#   - sensitivity_distribution   : spread of overall_cost across the random
#                                  combined study (run_sensitivity.py) vs. the
#                                  baseline.
#   - sensitivity_regression     : standardized regression coefficients of
#                                  overall_cost on the per-parameter factors of
#                                  the combined study (a statistical estimate of
#                                  the individual effects); also written as
#                                  regression_summary.json.
# Each figure covers both the decomposed solution (overall_cost) and the
# centralized/heuristic comparison (overall_cost_comparison).
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
TARGETS = [("overall_cost", "decomposed"), ("overall_cost_comparison", "comparison")]


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


def save(fig, export_path, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(export_path, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")


def _finite(values):
    return [v for v in values if isinstance(v, (int, float)) and np.isfinite(v)]


# ---------------------------------------------------------------- tornado
def plot_tornado(sweep, export_path):
    params = list(sweep.get("params", {}).keys())
    if not params:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.6 * len(params) + 2))
    for ax, (target, title) in zip(np.atleast_1d(axes), TARGETS):
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
    save(fig, export_path, "sensitivity_tornado")


# -------------------------------------------------------- response curves
def plot_response_curves(sweep, export_path):
    params = list(sweep.get("params", {}).keys())
    if not params:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharex=True)
    for ax, (target, title) in zip(np.atleast_1d(axes), TARGETS):
        base = sweep.get("baseline", {}).get(target)
        for p in params:
            f = sweep["params"][p].get("factors", [])
            c = sweep["params"][p].get(target, [])
            pts = [(x, y) for x, y in zip(f, c)
                   if isinstance(y, (int, float)) and np.isfinite(y)]
            if not pts:
                continue
            xs, ys = zip(*sorted(pts))
            ax.plot(xs, ys, marker="o", label=label(p))
        if base is not None:
            ax.axhline(base, color="k", linestyle=":", linewidth=1)
        ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("scale factor")
        ax.set_ylabel(r"Overall cost $C$")
        ax.set_title(title)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Cost response to each uncertainty")
    save(fig, export_path, "sensitivity_response_curves")


# ----------------------------------------------------------- distribution
def plot_distribution(summary, export_path):
    scenarios = summary.get("scenarios", [])
    oc = _finite([r.get("overall_cost") for r in scenarios])
    occ = _finite([r.get("overall_cost_comparison") for r in scenarios])
    if not oc and not occ:
        return
    base = summary.get("baseline", {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    data, labels, bases = [], [], []
    if oc:
        data.append(oc); labels.append("decomposed"); bases.append(base.get("overall_cost"))
    if occ:
        data.append(occ); labels.append("comparison"); bases.append(base.get("overall_cost_comparison"))
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
        ax2.hist(oc, bins=min(20, max(5, len(oc) // 2)), alpha=0.6, label="decomposed")
    if occ:
        ax2.hist(occ, bins=min(20, max(5, len(occ) // 2)), alpha=0.6, label="comparison")
    if base.get("overall_cost") is not None:
        ax2.axvline(base["overall_cost"], color="C0", linestyle="--", linewidth=1)
    if base.get("overall_cost_comparison") is not None:
        ax2.axvline(base["overall_cost_comparison"], color="C1", linestyle="--", linewidth=1)
    ax2.set_xlabel(r"Overall cost $C$")
    ax2.set_ylabel("count")
    ax2.legend(fontsize=8)
    ax2.set_title("Distribution")
    fig.suptitle("Sensitivity distribution (combined study)")
    save(fig, export_path, "sensitivity_distribution")


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


def plot_regression(csv_rows, export_path):
    if not csv_rows:
        return
    param_cols = [c for c in csv_rows[0].keys() if c.endswith("_factor_mean")]
    if not param_cols:
        print("  regression: no *_factor_mean columns in summary.csv; skipped")
        return
    out = {}
    for target, name in TARGETS:
        coef, r2, dropped = _standardized_regression(csv_rows, target, param_cols)
        out[name] = {"coefficients": coef, "r2": r2, "dropped_constant": dropped}

    params = sorted({p for v in out.values() for p in v["coefficients"]})
    if not params:
        print("  regression: all factor columns constant (use per_element=false "
              "in the combined study for a meaningful estimate); skipped figure")
        with open(os.path.join(export_path, "regression_summary.json"), "w") as f:
            json.dump(out, f, indent=4)
        return

    fig, ax = plt.subplots(figsize=(7, 3.5))
    width = 0.38
    xpos = np.arange(len(params))
    for i, (target, name) in enumerate(TARGETS):
        coef = out[name]["coefficients"]
        vals = [coef.get(p, 0.0) for p in params]
        r2 = out[name]["r2"]
        lbl = f"{name}" + (f" ($R^2$={r2:.2f})" if r2 is not None else "")
        ax.bar(xpos + (i - 0.5) * width, vals, width, label=lbl)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([label(p) for p in params])
    ax.set_ylabel("standardized regression coef.")
    ax.set_title("Estimated individual effects (combined study regression)")
    ax.legend(fontsize=8)
    save(fig, export_path, "sensitivity_regression")

    with open(os.path.join(export_path, "regression_summary.json"), "w") as f:
        json.dump(out, f, indent=4)
    print("  saved regression_summary.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_run", required=False, type=int, default=None)
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

    sweep = load_json(os.path.join(sens_dir, "scale_sweep_summary.json"))
    summary = load_json(os.path.join(sens_dir, "summary.json"))
    csv_rows = load_csv(os.path.join(sens_dir, "summary.csv"))

    print(f"Saving figures to {export_path}")
    if sweep:
        plot_tornado(sweep, export_path)
        plot_response_curves(sweep, export_path)
    else:
        print("  no scale_sweep_summary.json (run run_scale_sweep.py) -> "
              "skipping tornado + response curves")
    if summary:
        plot_distribution(summary, export_path)
    else:
        print("  no summary.json (run run_sensitivity.py) -> skipping distribution")
    if csv_rows:
        plot_regression(csv_rows, export_path)
    else:
        print("  no summary.csv -> skipping regression")


if __name__ == "__main__":
    main()
