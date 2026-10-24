"""Эксплораторно (К2): зависимость Δ, ρ_release, ρ_fold и отклонения H3 от mean-field от горизонта восстановления H_rec.

python analysis/explor_k2.py
H_rec = 1500, 3000, 6000 - из results/raw (база с репликами < 40 и варианты hrec3000/hrec6000 того же основного анализа,
готовые bootstrap-выборки results/analysis/t10_boot.npz); H_rec = 12000, 24000 - results/explor (теги x_k2, x_k2_ref).
Выход (results/analysis/):
  explor_k2_thresholds.csv  (E01) пороги и Δ по ячейкам и H_rec;
  explor_k2_drift.csv       (E02) наклон на удвоение H, приращения, экстраполяция до Δ = 0, сравнение log-модели и насыщения;
  explor_k2_h3.csv          (E03) отклонение ρ_release от mean-field (R1/FIFO/off/Exp), критерий |sim-MF| <= 0,10, H* нарушения;
  explor_k2_pb.csv          (E04) P_B(ρ) при разных H_rec (доля непрошедших восстановление);
  explor_k2_survival.csv    (E05) Каплан-Майер состояния отказа после перегрузки (B, H_rec = 24000) и интенсивность выхода;
  explor_k2_lifetime.csv    (E06) время жизни отказа τ(ρ), наклон ln τ по ρ и предсказанный сдвиг ρ_release на удвоение H.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explor_common import (  # noqa: E402
    HRECS, K2_CELLS, OUT, base_cfg, base_data, delta_row, explor_meta, load_explor, load_main, pct, rng_for, thr_of,
)
from main_hypotheses import mf_spinodals  # noqa: E402

T10 = pd.read_csv(OUT / "t10_sens.csv")
B10 = np.load(OUT / "t10_boot.npz", allow_pickle=True)
K10 = {k: i for i, k in enumerate(B10["key"])}
VAR = {1500: "base40", 3000: "hrec3000", 6000: "hrec6000"}
X = np.log2(np.array(HRECS) / 1500.0)  # 0..4: число удвоений


def collect() -> tuple[pd.DataFrame, dict]:
    """Таблица E01 и bootstrap-выборки (cell, H) -> (fold, rel)."""
    meta = explor_meta()
    ex = load_explor(["config_hash", "rho_test", "protocol", "failed", "replicate"], ["x_k2", "x_k2_ref"])
    rows, dr = [], {}
    for ci, (r, q, d, s) in enumerate(K2_CELLS):
        for hrec in HRECS:
            if hrec in VAR:
                j = K10[f"{VAR[hrec]}|{r}|{q}|{d}|{s}"]
                t = T10[(T10.variant == VAR[hrec]) & (T10.R == r) & (T10.D == d) & (T10.S == s) & (T10.Q == q)].iloc[0]
                row = {k: t[k] for k in ["fold", "fold_lo", "fold_hi", "fold_cens", "rel", "rel_lo", "rel_hi", "rel_cens",
                                         "delta", "d_lo", "d_hi", "N1", "N2"]}
                row["source"] = "results/raw (t10)"
                row["reps"] = int(t["reps"])
                fd, rd = B10["fold"][j], B10["rel"][j]
            else:
                h = meta[(meta.R == r) & (meta.Q == q) & (meta.D == d) & (meta.S == s) & (meta.h_rec == hrec)].config_hash
                if len(h) != 1:
                    raise RuntimeError("конфигурация explor не найдена")
                h = h.iloc[0]
                dd = ex[ex["config_hash"] == h]
                dd = dd[dd["rho_test"] <= 1.0 + 1e-9]
                a = thr_of(dd, "A", h, 10 + ci * 10 + 1)
                b = thr_of(dd, "B", h, 10 + ci * 10 + 2)
                row = delta_row(a, b)
                row["source"] = "results/explor (x_k2)"
                row["reps"] = int(dd["replicate"].max() + 1)
                fd, rd = a["draws"], b["draws"]
            row.update({"R": r, "Q": q, "D": d, "S": s, "H_rec": hrec})
            rows.append(row)
            dr[(r, q, d, s, hrec)] = (np.asarray(fd), np.asarray(rd))
    tab = pd.DataFrame(rows)
    tab = tab[["R", "Q", "D", "S", "H_rec", "reps", "source", "fold", "fold_lo", "fold_hi", "fold_cens", "rel", "rel_lo", "rel_hi",
               "rel_cens", "delta", "d_lo", "d_hi", "N1", "N2"]]
    return tab, dr


def ols_slope(y: np.ndarray, x: np.ndarray = X) -> np.ndarray:
    """Наклон МНК по оси x (число удвоений H); y: (..., 5)."""
    xc = x - x.mean()
    return (y * xc).sum(axis=-1) / (xc**2).sum()


def drift(tab: pd.DataFrame, dr: dict) -> pd.DataFrame:
    rows = []
    for (r, q, d, s) in K2_CELLS:
        for qty in ("rel", "fold", "delta"):
            draws = np.stack([(dr[(r, q, d, s, h)][1] if qty == "rel" else dr[(r, q, d, s, h)][0] if qty == "fold"
                               else dr[(r, q, d, s, h)][0] - dr[(r, q, d, s, h)][1]) for h in HRECS], axis=1)  # (B, 5)
            sub = tab[(tab.R == r) & (tab.D == d) & (tab.S == s)].sort_values("H_rec")
            y = sub[qty].to_numpy(float)
            cens = (sub["fold_cens"] if qty == "fold" else sub["rel_cens"] if qty == "rel" else sub["fold_cens"] + sub["rel_cens"]).tolist()
            b = ols_slope(draws)
            inc = np.diff(y)
            inc_dr = np.diff(draws, axis=1)
            row = {"R": r, "Q": q, "D": d, "S": s, "величина": qty, "censoring": ";".join(sorted(set(cens))),
                   "slope_per_doubling": float(ols_slope(y)), "slope_lo": pct(b)[0], "slope_hi": pct(b)[1]}
            for i in range(4):
                row[f"inc{i + 1}"] = float(inc[i])
                row[f"inc{i + 1}_lo"], row[f"inc{i + 1}_hi"] = pct(inc_dr[:, i])
            ratio = inc_dr[:, 3] / np.where(np.abs(inc_dr[:, 0]) > 1e-9, inc_dr[:, 0], np.nan)
            row["inc4_over_inc1"] = float(inc[3] / inc[0]) if abs(inc[0]) > 1e-9 else np.nan
            row["ratio_lo"], row["ratio_hi"] = (np.nanpercentile(ratio, [2.5, 97.5]) if np.isfinite(ratio).sum() > 100 else (np.nan, np.nan))
            # линейная по log2 H: R², остаток
            xc = X - X.mean()
            fit = y.mean() + ols_slope(y) * xc
            ss_res = float(((y - fit) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            row["R2_log"] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            row["max_resid_log"] = float(np.abs(y - fit).max())
            # насыщение: y = a - A exp(-k x), sigma = sd bootstrap
            sd = draws.std(axis=0, ddof=1)
            aic_lin = ss_res_w = float((((y - fit) / sd) ** 2).sum()) + 2 * 2
            row["AIC_log"] = aic_lin
            try:
                f = lambda x, a, A, k: a - A * np.exp(-k * x)  # noqa: E731
                sgn = np.sign(inc.sum()) or 1.0
                p0 = (y[-1] + sgn * abs(inc[-1]), sgn * abs(y[-1] - y[0]) + 1e-3, 0.3)
                popt, _ = curve_fit(f, X, y, p0=p0, sigma=sd, absolute_sigma=True, maxfev=20000, bounds=([-5, -5, 0.0], [5, 5, 5.0]))
                chi = float((((y - f(X, *popt)) / sd) ** 2).sum())
                row["sat_a_inf"], row["sat_k"], row["AIC_sat"] = float(popt[0]), float(popt[2]), chi + 2 * 3
            except Exception:  # noqa: BLE001
                row["sat_a_inf"], row["sat_k"], row["AIC_sat"] = np.nan, np.nan, np.nan
            # экстраполяция линейной log-модели до нуля (для Δ) на удвоениях от 1500
            if qty == "delta" and ols_slope(y) < 0:
                x0 = (y.mean() - ols_slope(y) * X.mean()) / (-ols_slope(y))  # y(x) = 0: a + s x = 0
                alpha = draws.mean(axis=1) - b * X.mean()
                x0_dr = np.where(b < 0, alpha / (-b), np.nan)
                row["H_zero"] = float(1500 * 2**x0)
                lo, hi = np.nanpercentile(x0_dr, [2.5, 97.5])
                row["H_zero_lo"], row["H_zero_hi"] = float(1500 * 2**lo), float(1500 * 2**hi)
            rows.append(row)
    return pd.DataFrame(rows)


def h3(tab: pd.DataFrame, dr: dict) -> pd.DataFrame:
    mf_rel, mf_fold = mf_spinodals(128, 10.0)
    r, q, d, s = K2_CELLS[0]
    rows = []
    devs = []
    for hrec in HRECS:
        rd = dr[(r, q, d, s, hrec)][1]
        t = tab[(tab.R == r) & (tab.D == d) & (tab.S == s) & (tab.H_rec == hrec)].iloc[0]
        dev = float(t.rel - mf_rel)
        dd = rd - mf_rel
        lo, hi = pct(dd)
        alo, ahi = pct(np.abs(dd))
        fd = dr[(r, q, d, s, hrec)][0] - mf_fold
        flo, fhi = pct(fd)
        rows.append({"H_rec": hrec, "mf_release": mf_rel, "sim_release": float(t.rel), "release_cens": t.rel_cens, "dev": dev,
                     "dev_lo": lo, "dev_hi": hi, "abs_dev_hi": ahi,
                     "H3_verdict": "H3 подтверждена" if ahi <= 0.10 else ("H3 опровергнута" if alo > 0.10 else "H3 неопределённо"),
                     "mf_fold": mf_fold, "sim_fold": float(t.fold), "fold_dev": float(t.fold - mf_fold), "fold_dev_lo": flo,
                     "fold_dev_hi": fhi})
        devs.append(dd)
    tab3 = pd.DataFrame(rows)
    dv = np.stack(devs, axis=1)  # (B,5)
    y = tab3["dev"].to_numpy()
    xc = X - X.mean()
    s_hat = float(ols_slope(y))
    a_hat = float(y.mean() - s_hat * X.mean())
    b = ols_slope(dv)
    a_dr = dv.mean(axis=1) - b * X.mean()
    # H*: уровень 0,10 в log-модели
    xs = (0.10 - a_hat) / s_hat
    xs_dr = np.where(b > 0, (0.10 - a_dr) / b, np.nan)
    lo, hi = np.nanpercentile(xs_dr, [2.5, 97.5])
    # H*, если дрейф затухает: по последним двум приращениям (оценка снизу не даётся)
    inc = np.diff(y)
    # прогноз log-модели при H = 12000, 24000 по данным только 1500-6000 (проверка предсказания К2)
    s3 = float(ols_slope(y[:3], X[:3]))
    a3 = float(y[:3].mean() - s3 * X[:3].mean())
    pred = {h: a3 + s3 * np.log2(h / 1500) for h in (12000, 24000)}
    out = pd.DataFrame([{"модель": "МНК по log2(H), 5 точек", "slope_per_doubling": s_hat, "intercept_1500": a_hat,
                         "H_star_dev010": float(1500 * 2**xs), "H_star_lo": float(1500 * 2**lo), "H_star_hi": float(1500 * 2**hi),
                         "inc1": inc[0], "inc2": inc[1], "inc3": inc[2], "inc4": inc[3],
                         "pred_12000_from_1500_6000": pred[12000], "pred_24000_from_1500_6000": pred[24000],
                         "obs_12000": y[3], "obs_24000": y[4]}])
    tab3 = tab3.merge(out.assign(H_rec=24000), on="H_rec", how="left")
    return tab3


def pb_table() -> pd.DataFrame:
    """P_B (доля отказов в окне W_last после перегрузки) и P_A на общих ρ для разных H_rec."""
    meta = explor_meta()
    ex = load_explor(["config_hash", "rho_test", "protocol", "failed", "replicate"], ["x_k2"])
    mm = load_main(["config_hash", "rho_test", "protocol", "failed", "replicate"], ["m_scr", "m_sens"])
    from main_common import cfg_meta
    cmeta = cfg_meta()
    rows = []
    for (r, q, d, s) in K2_CELLS:
        for hrec in HRECS:
            if hrec in VAR:
                v = VAR[hrec] if hrec != 1500 else "base"
                h = cmeta[(cmeta.variant == v) & (cmeta.R == r) & (cmeta.Q == q) & (cmeta.D == d) & (cmeta.S == s)].config_hash.iloc[0]
                x = mm[(mm.config_hash == h) & (mm.replicate < 40)]
            else:
                h = meta[(meta.R == r) & (meta.D == d) & (meta.S == s) & (meta.h_rec == hrec) & (meta.Q == q)].config_hash.iloc[0]
                x = ex[ex.config_hash == h]
            g = x.groupby(["protocol", "rho_test"]).agg(n=("failed", "size"), P=("failed", "mean")).reset_index()
            g.insert(0, "H_rec", hrec)
            g.insert(0, "S", s)
            g.insert(0, "D", d)
            g.insert(0, "R", r)
            rows.append(g)
    return pd.concat(rows, ignore_index=True)


def km_surv(t: np.ndarray, ev: np.ndarray, at: np.ndarray) -> np.ndarray:
    order = np.argsort(t, kind="stable")
    t, ev = t[order], ev[order]
    s = 1.0
    out = np.ones(len(at))
    times = np.unique(t[ev == 1])
    surv = []
    for u in times:
        at_risk = (t >= u).sum()
        d_i = ((t == u) & (ev == 1)).sum()
        s *= 1.0 - d_i / at_risk
        surv.append((u, s))
    for i, a in enumerate(at):
        val = 1.0
        for u, sv in surv:
            if u <= a:
                val = sv
            else:
                break
        out[i] = val
    return out


def survival() -> pd.DataFrame:
    """KM доли системы, ещё не вернувшейся к g >= 0,9, при протоколе B и H_rec = 24000 (цензурирование на горизонте)."""
    meta = explor_meta()
    ex = load_explor(["config_hash", "rho_test", "protocol", "recovery_time", "recovery_censored", "replicate"], ["x_k2"])
    at = np.array([1500.0, 3000.0, 6000.0, 12000.0, 24000.0 - 1e-6])
    rows = []
    for ci, (r, q, d, s) in enumerate(K2_CELLS):
        h = meta[(meta.R == r) & (meta.Q == q) & (meta.D == d) & (meta.S == s) & (meta.h_rec == 24000)].config_hash.iloc[0]
        x = ex[(ex.config_hash == h) & (ex.protocol == "B")]
        for rho in sorted(x.rho_test.unique()):
            if rho < 0.30 - 1e-9 or rho > 0.90 + 1e-9:
                continue
            xx = x[np.isclose(x.rho_test, rho)]
            t = xx.recovery_time.to_numpy(float)
            ev = (xx.recovery_censored.to_numpy() == 0).astype(int)
            s_hat = km_surv(t, ev, at)
            rng = rng_for(h, 500 + int(round(rho * 100)))
            bs = np.empty((400, len(at)))
            for b in range(400):
                i = rng.integers(0, len(t), len(t))
                bs[b] = km_surv(t[i], ev[i], at)
            row = {"R": r, "Q": q, "D": d, "S": s, "rho": rho, "n": len(t), "n_censored_24000": int((ev == 0).sum())}
            for k, hh in enumerate([1500, 3000, 6000, 12000, 24000]):
                row[f"S_{hh}"] = float(s_hat[k])
                row[f"S_{hh}_lo"], row[f"S_{hh}_hi"] = pct(bs[:, k])
            # интенсивность выхода (на 1000 ед. времени) на интервалах между удвоениями
            for k, (a, bb) in enumerate([(1500, 3000), (3000, 6000), (6000, 12000), (12000, 24000)]):
                i0, i1 = k, k + 1
                with np.errstate(divide="ignore", invalid="ignore"):
                    hz = -np.log(s_hat[i1] / s_hat[i0]) / (bb - a) * 1000 if s_hat[i0] > 0 and s_hat[i1] > 0 else np.nan
                    hzb = np.where((bs[:, i0] > 0) & (bs[:, i1] > 0), -np.log(bs[:, i1] / bs[:, i0]) / (bb - a) * 1000, np.nan)
                row[f"hazard_{a}_{bb}"] = float(hz) if np.isfinite(hz) else np.nan
                if np.isfinite(hzb).sum() > 50:
                    row[f"hazard_{a}_{bb}_lo"], row[f"hazard_{a}_{bb}_hi"] = np.nanpercentile(hzb, [2.5, 97.5])
            rows.append(row)
    return pd.DataFrame(rows)


def lifetime() -> pd.DataFrame:
    """Экспоненциальная оценка времени жизни отказа после перегрузки: τ(ρ) = Σ min(t, H)/число выходов (B, H_rec = 24000),
    затем наклон ln τ по ρ и предсказанный сдвиг ρ_release на удвоение горизонта = ln 2 / наклон."""
    meta = explor_meta()
    ex = load_explor(["config_hash", "rho_test", "protocol", "recovery_time", "recovery_censored", "replicate"], ["x_k2"])
    ranges = {K2_CELLS[0]: (0.30, 0.40), K2_CELLS[1]: (0.70, 0.85), K2_CELLS[2]: (0.55, 0.65)}
    rows = []
    for (r, q, d, s) in K2_CELLS:
        h = meta[(meta.R == r) & (meta.Q == q) & (meta.D == d) & (meta.S == s) & (meta.h_rec == 24000)].config_hash.iloc[0]
        x = ex[(ex.config_hash == h) & (ex.protocol == "B")]
        lo_r, hi_r = ranges[(r, q, d, s)]
        rhos = [rho for rho in sorted(x.rho_test.unique()) if lo_r - 1e-9 <= rho <= hi_r + 1e-9]
        data = {rho: x[np.isclose(x.rho_test, rho)] for rho in rhos}
        rng = rng_for(h, 900)

        def fit(sample: dict) -> tuple[float, float, list]:
            lt = []
            for rho in rhos:
                t = sample[rho]["recovery_time"].to_numpy(float)
                ev = (sample[rho]["recovery_censored"].to_numpy() == 0).sum()
                lt.append((rho, float(t.sum() / ev) if ev > 0 else np.nan, int(ev)))
            xs = np.array([a for a, b_, _ in lt if np.isfinite(b_)])
            ys = np.log([b_ for _, b_, _ in lt if np.isfinite(b_)])
            if len(xs) < 2:
                return np.nan, np.nan, lt
            sl, ic = np.polyfit(xs, ys, 1)
            return float(sl), float(ic), lt

        sl, ic, lt = fit(data)
        bs = []
        for _ in range(1000):
            smp = {rho: data[rho].iloc[rng.integers(0, len(data[rho]), len(data[rho]))] for rho in rhos}
            bs.append(fit(smp)[0])
        bs = np.array(bs)
        bs = bs[np.isfinite(bs) & (bs > 0)]
        for rho, tau, ev in lt:
            rows.append({"R": r, "Q": q, "D": d, "S": s, "rho": rho, "events": ev, "n": 40, "tau_hat": tau, "kind": "точка"})
        rows.append({"R": r, "Q": q, "D": d, "S": s, "rho": np.nan, "events": np.nan, "n": np.nan, "tau_hat": np.nan, "kind": "подгонка",
                     "slope_ln_tau_per_rho": sl, "slope_lo": pct(bs)[0], "slope_hi": pct(bs)[1],
                     "pred_shift_per_doubling": float(np.log(2) / sl), "pred_shift_lo": float(np.log(2) / pct(bs)[1]),
                     "pred_shift_hi": float(np.log(2) / pct(bs)[0]), "rho_range": f"{lo_r}-{hi_r}"})
    return pd.DataFrame(rows)


def main() -> int:
    tab, dr = collect()
    tab.to_csv(OUT / "explor_k2_thresholds.csv", index=False, encoding="utf-8")
    print(tab[["R", "D", "S", "H_rec", "fold", "fold_cens", "rel", "rel_cens", "delta", "d_lo", "d_hi"]].round(3).to_string())
    dft = drift(tab, dr)
    dft.to_csv(OUT / "explor_k2_drift.csv", index=False, encoding="utf-8")
    print(dft[["R", "D", "S", "величина", "slope_per_doubling", "slope_lo", "slope_hi", "inc1", "inc2", "inc3", "inc4", "R2_log",
               "AIC_log", "AIC_sat", "sat_a_inf", "sat_k"]].round(4).to_string())
    h3t = h3(tab, dr)
    h3t.to_csv(OUT / "explor_k2_h3.csv", index=False, encoding="utf-8")
    print(h3t.round(4).to_string())
    pbt = pb_table()
    pbt.to_csv(OUT / "explor_k2_pb.csv", index=False, encoding="utf-8")
    sv = survival()
    sv.to_csv(OUT / "explor_k2_survival.csv", index=False, encoding="utf-8")
    lf = lifetime()
    lf.to_csv(OUT / "explor_k2_lifetime.csv", index=False, encoding="utf-8")
    print(lf.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
