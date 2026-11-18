"""Compare alert policies on the best detector (seasonal) across all 17 AWS series.

  fixed            threshold = factor x max score in the probationary period (the v0.1 rule)
  fixed + 2 pts    same, but the alarm must hold for 2 consecutive points (10 min)
  adaptive         threshold = factor x 99.9th percentile of the previous 7 days
  adaptive + 2 pts adaptive, with the 2-point persistence filter

Usage:  python -m capfc.policies
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from capfc import RESULTS_DIR
from capfc.data import load_all
from capfc.detect import adaptive_alarms, alarms, evaluate, persistent, seasonal_detector

FACTORS = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0)
POLICIES = {
    "fixed (v0.1)": lambda sc, f: alarms(sc, f),
    "fixed + 2 pts": lambda sc, f: persistent(alarms(sc, f), 2),
    "adaptive": lambda sc, f: adaptive_alarms(sc, f),
    "adaptive + 2 pts": lambda sc, f: persistent(adaptive_alarms(sc, f), 2),
}
STYLE = {"fixed (v0.1)": ("#9a9a9a", "o"), "fixed + 2 pts": ("#4C78A8", "s"),
         "adaptive": ("#F58518", "^"), "adaptive + 2 pts": ("#E45756", "D")}


def sweep(data) -> pd.DataFrame:
    rows = []
    for name, (s, windows) in data.items():
        score = seasonal_detector(s)
        for pol, fn in POLICIES.items():
            for f in FACTORS:
                rows.append({"policy": pol, "factor": f, **evaluate(fn(score, f), windows)})
    sw = pd.DataFrame(rows).groupby(["policy", "factor"]).sum().reset_index()
    sw["detection_rate"] = sw["detected"] / sw["windows"]
    sw["false_alarms_per_week"] = sw["false_alarm_events"] / sw["days"] * 7
    return sw


def at_budget(sw: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Best detection rate each policy reaches without exceeding `budget` false alarms/week."""
    ok = sw[sw["false_alarms_per_week"] <= budget]
    best = ok.loc[ok.groupby("policy")["detection_rate"].idxmax()]
    return best.set_index("policy")[["factor", "detection_rate", "false_alarms_per_week"]]


def chart(sw: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for pol, g in sw.groupby("policy"):
        g = g.sort_values("false_alarms_per_week")
        c, m = STYLE[pol]
        ax.plot(g["false_alarms_per_week"], g["detection_rate"] * 100, marker=m, color=c,
                label=pol, lw=2 if pol != "fixed (v0.1)" else 1.2, ms=5)
    ax.axvline(2, color="#555", ls=":", lw=1)
    ax.text(2.1, 3, "budget: 2 false\npages/metric/week", fontsize=8, color="#555")
    ax.set_xlabel("false alarms per metric per week (event level)")
    ax.set_ylabel("incidents detected (%)")
    ax.set_title("Same detector, different alert policy (seasonal detector, 17 AWS metrics)", loc="left")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 100)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "policy_tradeoff.png", dpi=160)


def main() -> None:
    sw = sweep(load_all())
    sw.to_csv(RESULTS_DIR / "policy_tradeoff.csv", index=False, float_format="%.3f")
    rows = []
    for budget in (1, 2, 4):
        b = at_budget(sw, budget)
        b.insert(0, "budget_per_week", budget)
        rows.append(b.reset_index())
    table = pd.concat(rows)
    table.to_csv(RESULTS_DIR / "policy_at_budget.csv", index=False, float_format="%.3f")
    print(table.round(3).to_string(index=False))
    chart(sw)


if __name__ == "__main__":
    main()
