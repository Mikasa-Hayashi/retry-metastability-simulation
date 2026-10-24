"""График ветвей mean-field модели R1 (figures/theory_branches.*) и данные рядом."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import theory as th

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rhos = np.round(np.arange(0.10, 1.0001, 0.005), 4)
    rows: list[tuple[str, float, float, int, float]] = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for k_queue, label, ls in [(128, "K=128 (очередь)", "-"), (None, "K=inf", ":")]:
        p = th.MFParams(k_queue=k_queue)
        for rho in rhos:
            for fp in th.fixed_points(float(rho), p, n_grid=1201):
                rows.append((label, float(rho), fp.q, int(fp.stable), fp.slope))
        for stable in (1, 0):
            pts = [(r, q) for (lb, r, q, st, _) in rows if lb == label and st == stable]
            if pts:
                x, y = zip(*pts, strict=True)
                axes[0].scatter(
                    x,
                    y,
                    s=6 if stable else 4,
                    marker="o" if stable else "x",
                    color="C0" if k_queue else "C1",
                    label=f"{label}, {'устойч.' if stable else 'неустойч.'}",
                )
    axes[0].set_xlabel("ρ")
    axes[0].set_ylabel("q (вероятность неуспеха попытки)")
    axes[0].set_title("Mean-field R1: неподвижные точки q = F(q; ρ)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    csv_path = ROOT / "figures" / "theory_check_protocols.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as fh:
            data = list(csv.DictReader(fh))
        for trig, col in [("4.0", "C2"), ("2.0", "C3")]:
            sub = [d for d in data if d["trigger"] == trig]
            r = [float(d["rho"]) for d in sub]
            axes[1].plot(
                r,
                [float(d["P_A"]) for d in sub],
                "o-",
                color=col,
                ms=3,
                label=f"P_A, триггер ×{trig[0]}",
            )
        sub = [d for d in data if d["trigger"] == "4.0"]
        axes[1].plot(
            [float(d["rho"]) for d in sub],
            [float(d["P_B"]) for d in sub],
            "ks--",
            ms=3,
            label="P_B",
        )
    axes[1].axvline(0.277, color="C0", lw=0.8, ls="--")
    axes[1].axvline(0.928, color="C0", lw=0.8, ls="--")
    axes[1].set_xlabel("ρ")
    axes[1].set_ylabel("доля отказов")
    axes[1].set_title("Прототип протоколов A/B (независимый мини-симулятор)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = ROOT / "figures"
    fig.savefig(out / "theory_branches.png", dpi=150)
    fig.savefig(out / "theory_branches.svg")
    with (out / "theory_branches.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "rho", "q", "stable", "slope"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
