#!/usr/bin/env python3
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats

DISTS = {
    "gumbel": {"dist": stats.gumbel_r,   "names": ("loc", "scale")},
    "gev":    {"dist": stats.genextreme, "names": ("shape", "loc", "scale")},
}


def read_numbers(path):
    if path is None:
        stream = sys.stdin
    else:
        stream = open(path, "r", encoding="utf-8")
    vals = []
    with stream as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.replace(",", " ")
            for tok in line.split():
                vals.append(float(tok))
    x = np.asarray(vals, dtype=float)
    return x[np.isfinite(x)]


def fit_distribution(x, name):
    dist_class, names = DISTS[name]["dist"], DISTS[name]["names"]
    params = dist_class.fit(x)
    frozen = dist_class(*params)
    n = x.size
    param_str = ", ".join(f"{nm}={v:.6g}" for nm, v in zip(names, params))
    xs = np.linspace(x.min(), x.max(), 600)
    x_sorted = np.sort(x)
    surv_emp = 1 - (np.arange(1, n + 1) - 1) / n
    return {
        "frozen": frozen,
        "param_str": param_str,
        "xs": xs,
        "x_sorted": x_sorted,
        "surv_emp": surv_emp,
        "n": n,
    }


def plot_survival(ax, data, dist_name, xlabel):
    frozen = data["frozen"]
    ax.plot(data["x_sorted"], data["surv_emp"], marker=".", linestyle="none", label="observed")
    ax.plot(data["xs"], frozen.sf(data["xs"]), linewidth=2, label="expected")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cumulative frequency")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v * data['n']:.0f}"))
    ax.legend()

    # --- hidden for now: histogram + fitted PDF ---
    # pdf = frozen.pdf(data["xs"])
    # ax.hist(x, bins="fd", density=True, alpha=0.6)
    # ax.plot(xs, pdf, linewidth=2)
    # ax.set_title("Histogram + fitted PDF")

    # --- hidden for now: Q-Q plot ---
    # p = (np.arange(1, n + 1) - 0.5) / n
    # theo_q = frozen.ppf(p)
    # ax.scatter(theo_q, x_sorted, s=14)
    # q1x, q3x = np.quantile(theo_q, [0.25, 0.75])
    # q1y, q3y = np.quantile(x_sorted, [0.25, 0.75])
    # slope = (q3y - q1y) / (q3x - q1x)
    # intercept = q1y - slope * q1x
    # line_x = np.array([theo_q.min(), theo_q.max()])
    # ax.plot(line_x, intercept + slope * line_x, linewidth=2)
    # ax.set_title("Q-Q plot")


FB_XLABEL = "Forward-backward score"
F_XLABEL = "Backward score"


def main():
    parser = argparse.ArgumentParser(
        description="Fit Gumbel/GEV to full-mid (Forward-backward) and filter-all (Backward) "
                    "scores; one --input per family, all families in a single figure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", dest="inputs", action="append", default=[], metavar="LABEL:FULL:FILTER",
                        help="One per family: label plus full-mid and filter-all score files "
                             "(empty path = missing side). Repeat for each family.")
    parser.add_argument("--fit", type=str, choices=sorted(DISTS), default="gumbel", help="Distribution to fit")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    if not args.inputs:
        parser.error("at least one --input LABEL:FULL:FILTER is required")

    families = []
    for spec in args.inputs:
        if spec.count(":") < 2:
            parser.error(f"bad --input {spec!r}: expected LABEL:FULL:FILTER")
        label, full_path, filter_path = spec.split(":", 2)
        if not label:
            parser.error(f"bad --input {spec!r}: empty label")
        x_full = read_numbers(full_path) if full_path else np.asarray([], dtype=float)
        x_filter = read_numbers(filter_path) if filter_path else np.asarray([], dtype=float)
        has_full = x_full.size >= 5
        has_filter = x_filter.size >= 5
        if not has_full and not has_filter:
            print(f"# Skipping {label}: only {x_full.size} full and {x_filter.size} filter finite data points")
            continue
        if not has_full:
            print(f"# {label}: only {x_full.size} full finite data points (panel left empty)")
            x_full = None
        if not has_filter:
            print(f"# {label}: only {x_filter.size} filter finite data points (panel left empty)")
            x_filter = None
        families.append((label, x_full, x_filter))
    if not families:
        sys.exit(1)

    fitted = [(label,
               fit_distribution(x_full, args.fit) if x_full is not None else None,
               fit_distribution(x_filter, args.fit) if x_filter is not None else None)
              for label, x_full, x_filter in families]

    n_fams = len(fitted)
    fig, axes = plt.subplots(n_fams, 2, sharex="col", sharey=False,
                             figsize=(10, 5 * n_fams), squeeze=False)
    for row, (label, full_data, filter_data) in enumerate(fitted):
        ax_l, ax_r = axes[row, 0], axes[row, 1]
        if full_data is not None:
            plot_survival(ax_l, full_data, args.fit, FB_XLABEL)
            ax_l.set_title(f"{label} (Forward-backward)\n{full_data['param_str']}")
            print(f"Fitted {args.fit} ({label} full): {full_data['param_str']}")
        else:
            ax_l.set_xlabel(FB_XLABEL)
            ax_l.set_ylabel("cumulative frequency")
            ax_l.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax_l.transAxes)
            ax_l.set_title(f"{label} (Forward-backward)")
        if filter_data is not None:
            plot_survival(ax_r, filter_data, args.fit, F_XLABEL)
            ax_r.set_title(f"{label} (Backward)\n{filter_data['param_str']}")
            print(f"Fitted {args.fit} ({label} filter): {filter_data['param_str']}")
        else:
            ax_r.set_xlabel(F_XLABEL)
            ax_r.set_ylabel("cumulative frequency")
            ax_r.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax_r.transAxes)
            ax_r.set_title(f"{label} (Backward)")

    fig.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
