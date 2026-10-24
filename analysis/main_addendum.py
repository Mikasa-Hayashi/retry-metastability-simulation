"""Дополнение к основному анализу (по замечаниям проверки К1, В4, В9): эксплораторные диагностики по данным results/raw/.

python analysis/main_addendum.py     (после main_thresholds.py и main_hypotheses.py)

Все расчёты ведутся по g_final основных прогонов (теги m_scr, m_ref1, m_r100; ρ ≤ 1; 100 реплик на точку), новых прогонов нет.
Это эксплораторный анализ вне препрегистрации; подтверждающие результаты (T01-T24) не меняются.

Выход (results/analysis/):
  t40_bimodality.csv      К1: диагностика бимодальности g_final, 12 активных ячеек (T40)
  t40b_hist.csv           данные гистограмм для F14
  t40c_gap_share72.csv    К1: доля g_final в [0,3; 0,7] по всем 72 ячейкам, протоколы A и B (T40c)
  t41_theta_delta.csv     К1: Δ при θ = 0,3 / 0,5 / 0,7 по 12 ячейкам, пересчёт g_final (T41); t41b_theta_h1p.csv: H1′ при каждом θ
  t41c_theta_point72.csv  К1: точечная проверка по всем 72 ячейкам при θ = 0,3 и 0,7
  t42_gap_delta.csv       К1: Δ_gap (классификация по разрыву g_final, не по θ) по 12 ячейкам (T42); t42b_gap_h1p.csv: H1′ на этой мере
  t43_h1_rules.csv        В4: H1 по двум правилам (T43)
  t44_methods.csv         В9: ρ_fold, ρ_release, Δ тремя методами (логистика/изотоника по плану, линейная интерполяция сырых P) (T44)
  t44b_r2_on_exp.csv      В9: сырые P_A вокруг порога R2/FIFO/on/Exp
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_common import ACTIVE12, BASE_TAGS, OUT, SS, base_meta, load_main, pct, rng_for  # noqa: E402
from src.thresholds import _isotonic_threshold, aggregate, bootstrap_thresholds, point_threshold  # noqa: E402

THETAS = [0.3, 0.5, 0.7]
BOOT_A = 2000
EDGES = np.linspace(0, 1, 41)  # шаг 0,025


def thr_draw(rho: np.ndarray, failed: np.ndarray, h: str, code: int, n_boot: int = BOOT_A) -> dict:
    x, n, k = aggregate(rho, failed)
    val, cens, method = point_threshold(x, n, k, 0.5, 1.0)
    draws = bootstrap_thresholds(x, n, k, rng_for(h, code), n_boot, 0.5, 1.0)
    return {"value": float(val), "cens": cens, "draws": draws}


def work_delta(args: tuple) -> dict:
    """Δ при заданном правиле отказа (булев массив failed уже посчитан)."""
    key, h, ra, fa, rb, fb, code = args
    a = thr_draw(ra, fa, h, code + 1)
    b = thr_draw(rb, fb, h, code + 2)
    d = a["draws"] - b["draws"]
    lo, hi = pct(d)
    return {"key": key, "fold": a["value"], "fold_cens": a["cens"], "rel": b["value"], "rel_cens": b["cens"],
            "delta": a["value"] - b["value"], "d_lo": lo, "d_hi": hi, "fd": a["draws"], "rd": b["draws"]}


def h1p_from(cells: dict[tuple, dict], r_list=("R1", "R2")) -> dict:
    """r(S) = Δ_on/Δ_off (R1, R2 усреднены), разность r(Exp) − r(LogN) с bootstrap-ДИ; cells[(R, D, S)] -> результат work_delta."""
    out: dict = {}
    rr: dict[str, np.ndarray] = {}
    pt: dict[str, float] = {}
    for s in SS:
        on = sum(cells[(r, "on", s)]["fd"] - cells[(r, "on", s)]["rd"] for r in r_list)
        off = sum(cells[(r, "off", s)]["fd"] - cells[(r, "off", s)]["rd"] for r in r_list)
        with np.errstate(divide="ignore", invalid="ignore"):
            rr[s] = np.where(off > 1e-9, on / off, np.nan)
        on_p = sum(cells[(r, "on", s)]["delta"] for r in r_list)
        off_p = sum(cells[(r, "off", s)]["delta"] for r in r_list)
        pt[s] = on_p / off_p if off_p > 1e-9 else float("nan")
        v = rr[s][~np.isnan(rr[s])]
        out[f"r_{s}"] = pt[s]
        out[f"r_{s}_lo"], out[f"r_{s}_hi"] = pct(v) if len(v) > 100 else (np.nan, np.nan)
    diff = rr["Exp"] - rr["LogN"]
    ok = ~np.isnan(diff)
    out["diff"] = pt["Exp"] - pt["LogN"]
    out["diff_lo"], out["diff_hi"] = pct(diff[ok]) if ok.sum() > 100 else (np.nan, np.nan)
    out["valid_draws"] = int(ok.sum())
    return out


def bimodality(g: np.ndarray) -> dict:
    """Диагностика двух режимов по g_final пула прогонов ячейки."""
    cnt, _ = np.histogram(g, bins=EDGES)
    dens = gaussian_filter1d(cnt.astype(float), 1.0)
    ctr = (EDGES[:-1] + EDGES[1:]) / 2
    lo_m, hi_m = ctr < 0.65, ctr >= 0.65
    i_lo = int(np.argmax(np.where(lo_m, dens, -1)))
    i_hi = int(np.argmax(np.where(hi_m, dens, -1)))
    seg = dens[i_lo:i_hi + 1]
    plateau = np.flatnonzero(seg <= seg.min() + 0.5) + i_lo  # плато минимальной плотности (в пределах 0,5 прогона сглаженной гистограммы)
    i_v = int(plateau[len(plateau) // 2])  # середина плато: при пустом зазоре точка не выбирается произвольно
    cut = float(ctr[i_v])
    ratio = float(dens[i_v] / min(dens[i_lo], dens[i_hi])) if min(dens[i_lo], dens[i_hi]) > 0 else float("nan")
    share_low = float(np.mean(g < cut))
    # смесь двух нормалей (g на [0, 1]; reg_covar для вырожденного режима g = 0)
    x = g.reshape(-1, 1)
    g1 = GaussianMixture(1, reg_covar=1e-4, random_state=0).fit(x)
    g2 = GaussianMixture(2, reg_covar=1e-4, random_state=0, n_init=3).fit(x)
    mu = g2.means_.ravel()
    sd = np.sqrt(g2.covariances_.ravel())
    ashman = float(np.sqrt(2) * abs(mu[0] - mu[1]) / np.sqrt(sd[0] ** 2 + sd[1] ** 2))
    w = g2.weights_
    order = np.argsort(mu)
    # коэффициент бимодальности Сарла
    n = len(g)
    m3 = float(np.mean((g - g.mean()) ** 3) / g.std() ** 3)
    m4 = float(np.mean((g - g.mean()) ** 4) / g.std() ** 4 - 3.0)
    bc = (m3 ** 2 + 1) / (m4 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    gap = bool((ratio <= 0.25) and (share_low >= 0.03) and (ctr[i_lo] < 0.6))
    return {"valley_cut": cut, "peak_low": float(ctr[i_lo]), "peak_high": float(ctr[i_hi]), "valley_ratio": ratio, "share_low_mode": share_low,
            "share_gap_0.3_0.7": float(np.mean((g >= 0.3) & (g <= 0.7))),
            "gmm_mu_low": float(mu[order[0]]), "gmm_mu_high": float(mu[order[1]]), "gmm_w_low": float(w[order[0]]),
            "ashman_D": ashman, "dBIC_1_minus_2": float(g1.bic(x) - g2.bic(x)), "sarle_BC": float(bc), "two_modes": gap}


def main() -> int:
    meta = base_meta()
    t01 = pd.read_csv(OUT / "t01_thresholds72.csv")
    res = load_main(["config_hash", "rho_test", "protocol", "failed", "g_final", "replicate", "tag"], BASE_TAGS)
    res = res[res["rho_test"] <= 1.0 + 1e-9]
    assert (res["failed"] == (res["g_final"] < 0.5)).all()
    by_h = {h: d for h, d in res.groupby("config_hash")}
    hmap = {(r.R, r.Q, r.D, r.S): r.config_hash for r in meta.itertuples()}

    # ---- К1.1 бимодальность g_final, 12 активных ячеек ----
    rows, hist = [], []
    for (r, q, d, s) in ACTIVE12:
        h = hmap[(r, q, d, s)]
        x = by_h[h]
        trow = t01[t01.config_hash == h].iloc[0]
        inw = x[(x.rho_test >= trow.rel) & (x.rho_test <= trow.fold)]  # окно между порогами
        g_all = x["g_final"].to_numpy()
        info = bimodality(g_all)
        row = {"R": r, "D": d, "S": s, "n_runs": len(x), **info}
        for pr in ("A", "B"):
            gp = x[x.protocol == pr]["g_final"].to_numpy()
            row[f"share_0.3_0.7_{pr}"] = float(np.mean((gp >= 0.3) & (gp <= 0.7)))
        gw = inw["g_final"].to_numpy()
        row["n_window"] = len(inw)
        row["share_0.3_0.7_window"] = float(np.mean((gw >= 0.3) & (gw <= 0.7))) if len(gw) else np.nan
        row["window_rho"] = f"{trow.rel:.3f}-{trow.fold:.3f}"
        if len(gw) > 50:
            iw = bimodality(gw)
            row["window_ashman_D"] = iw["ashman_D"]
            row["window_valley_ratio"] = iw["valley_ratio"]
            row["window_two_modes"] = iw["two_modes"]
        rows.append(row)
        cnt, _ = np.histogram(g_all, bins=EDGES)
        for k in range(len(cnt)):
            hist.append({"R": r, "D": d, "S": s, "bin_lo": EDGES[k], "bin_hi": EDGES[k + 1], "count": int(cnt[k]), "valley_cut": info["valley_cut"]})
    bm = pd.DataFrame(rows)
    bm.to_csv(OUT / "t40_bimodality.csv", index=False, encoding="utf-8")
    pd.DataFrame(hist).to_csv(OUT / "t40b_hist.csv", index=False, encoding="utf-8")
    print(bm.round(3).to_string())

    # доля в [0,3; 0,7] по всем 72 ячейкам
    r72 = []
    for m in meta.itertuples():
        x = by_h[m.config_hash]
        row = {"R": m.R, "Q": m.Q, "D": m.D, "S": m.S}
        for pr in ("A", "B"):
            gp = x[x.protocol == pr]["g_final"].to_numpy()
            row[f"share_0.3_0.7_{pr}"] = float(np.mean((gp >= 0.3) & (gp <= 0.7)))
            row[f"share_fail_{pr}"] = float(np.mean(gp < 0.5))
        r72.append(row)
    pd.DataFrame(r72).to_csv(OUT / "t40c_gap_share72.csv", index=False, encoding="utf-8")

    # ---- К1.2 θ-чувствительность (пересчёт g_final), 12 ячеек ----
    jobs = []
    for ti, th in enumerate(THETAS):
        for (r, q, d, s) in ACTIVE12:
            h = hmap[(r, q, d, s)]
            x = by_h[h]
            a, b = x[x.protocol == "A"], x[x.protocol == "B"]
            code = 10 if th == 0.5 else 100 * (ti + 1)  # θ = 0,5: те же коды, что в main_thresholds -> воспроизводит T01
            jobs.append(((th, r, q, d, s), h, a["rho_test"].to_numpy(), (a["g_final"].to_numpy() < th).astype(int),
                         b["rho_test"].to_numpy(), (b["g_final"].to_numpy() < th).astype(int), code))
    with mp.Pool(7) as pool:
        out = pool.map(work_delta, jobs, chunksize=1)
    om = {o["key"]: o for o in out}
    trows, hrows = [], []
    for o in out:
        th, r, q, d, s = o["key"]
        trows.append({"theta": th, "R": r, "D": d, "S": s, "fold": o["fold"], "fold_cens": o["fold_cens"], "rel": o["rel"], "rel_cens": o["rel_cens"],
                      "delta": o["delta"], "d_lo": o["d_lo"], "d_hi": o["d_hi"]})
    th_tab = pd.DataFrame(trows)
    th_tab.to_csv(OUT / "t41_theta_delta.csv", index=False, encoding="utf-8")
    for th in THETAS:
        cells = {(r, d, s): om[(th, r, q, d, s)] for (r, q, d, s) in ACTIVE12}
        hp = h1p_from(cells)
        hrows.append({"theta": th, **hp})
    hp_tab = pd.DataFrame(hrows)
    hp_tab.to_csv(OUT / "t41b_theta_h1p.csv", index=False, encoding="utf-8")
    print(th_tab.round(3).to_string())
    print(hp_tab.round(3).to_string())
    # сверка θ = 0,5 с T01
    chk = th_tab[th_tab.theta == 0.5].merge(t01[["R", "Q", "D", "S", "delta"]].query("Q == 'FIFO'"), on=["R", "D", "S"], suffixes=("", "_t01"))
    print("сверка θ = 0,5 с T01: max |Δ − Δ_T01| =", float((chk.delta - chk.delta_t01).abs().max()))

    # точечная проверка по 72 ячейкам
    p72 = []
    for m in meta.itertuples():
        x = by_h[m.config_hash]
        a, b = x[x.protocol == "A"], x[x.protocol == "B"]
        row = {"R": m.R, "Q": m.Q, "D": m.D, "S": m.S}
        for th in (0.3, 0.5, 0.7):
            vf, cf, _ = point_threshold(*aggregate(a["rho_test"].to_numpy(), (a["g_final"].to_numpy() < th).astype(int)), 0.5, 1.0)
            vr, cr, _ = point_threshold(*aggregate(b["rho_test"].to_numpy(), (b["g_final"].to_numpy() < th).astype(int)), 0.5, 1.0)
            row[f"delta_{th}"] = vf - vr
            row[f"fold_cens_{th}"] = cf
        p72.append(row)
    p72 = pd.DataFrame(p72)
    p72.to_csv(OUT / "t41c_theta_point72.csv", index=False, encoding="utf-8")
    act = {(r, q, d, s) for (r, q, d, s) in ACTIVE12}
    extra = p72[~p72.apply(lambda z: (z.R, z.Q, z.D, z.S) in act, axis=1)]
    print("вне 12 активных: максимум Δ при θ=0,3 / 0,5 / 0,7:", extra["delta_0.3"].max(), extra["delta_0.5"].max(), extra["delta_0.7"].max())
    print(extra[(extra["delta_0.3"] > 0.02) | (extra["delta_0.7"] > 0.02)].round(3).to_string())

    # ---- К1.3 Δ_gap: отказ = g_final ниже порога-долины ячейки (разрыв между режимами), не θ ----
    jobs, gaprow = [], {}
    for ci, (r, q, d, s) in enumerate(ACTIVE12):
        h = hmap[(r, q, d, s)]
        x = by_h[h]
        info = bm[(bm.R == r) & (bm.D == d) & (bm.S == s)].iloc[0]
        gaprow[(r, d, s)] = info
        cut = float(info.valley_cut)
        a, b = x[x.protocol == "A"], x[x.protocol == "B"]
        jobs.append(((r, q, d, s), h, a["rho_test"].to_numpy(), (a["g_final"].to_numpy() < cut).astype(int),
                     b["rho_test"].to_numpy(), (b["g_final"].to_numpy() < cut).astype(int), 700 + 10 * ci))
    with mp.Pool(7) as pool:
        out = pool.map(work_delta, jobs, chunksize=1)
    gm = {o["key"]: o for o in out}
    grows = []
    cells_gap = {}
    for (r, q, d, s) in ACTIVE12:
        o = gm[(r, q, d, s)]
        info = gaprow[(r, d, s)]
        # ячейка без разрыва: Δ_gap = 0 по соглашению (два режима не различимы)
        if not info.two_modes:
            o = {**o, "delta": 0.0, "d_lo": 0.0, "d_hi": 0.0, "fd": np.zeros_like(o["fd"]), "rd": np.zeros_like(o["rd"])}
        cells_gap[(r, d, s)] = o
        cut_ = float(info.valley_cut)
        xa_, xb_ = x[x.protocol == "A"], x[x.protocol == "B"]
        wlo, whi = gm[(r, q, d, s)]["rel"], gm[(r, q, d, s)]["fold"]
        wa = xa_[(xa_.rho_test > wlo) & (xa_.rho_test < whi)]["g_final"].to_numpy()
        wb = xb_[(xb_.rho_test > wlo) & (xb_.rho_test < whi)]["g_final"].to_numpy()
        grows.append({"R": r, "D": d, "S": s, "valley_cut": info.valley_cut,
                      "win_n_A": len(wa), "win_share_low_A": float(np.mean(wa < cut_)) if len(wa) else np.nan,
                      "win_share_low_B": float(np.mean(wb < cut_)) if len(wb) else np.nan, "two_modes": bool(info.two_modes), "valley_ratio": info.valley_ratio,
                      "fold_gap": gm[(r, q, d, s)]["fold"], "rel_gap": gm[(r, q, d, s)]["rel"], "delta_gap": o["delta"], "d_lo": o["d_lo"], "d_hi": o["d_hi"],
                      "delta_theta05": float(t01[(t01.R == r) & (t01.Q == q) & (t01.D == d) & (t01.S == s)].delta.iloc[0])})
    gap_tab = pd.DataFrame(grows)
    gap_tab.to_csv(OUT / "t42_gap_delta.csv", index=False, encoding="utf-8")
    gh = h1p_from(cells_gap)
    pd.DataFrame([{"measure": "Δ_gap (порог-долина)", **gh}]).to_csv(OUT / "t42b_gap_h1p.csv", index=False, encoding="utf-8")
    print(gap_tab.round(3).to_string())
    print(gh)

    # ---- В4 H1: два правила ----
    h1 = pd.read_csv(OUT / "t20_h1.csv")
    v = h1[h1["set"].str.startswith("18")].copy()
    v["verdict_cells"] = np.where(v.cell_lo >= 0.8, "отвергнута (нижняя граница ≥ 0,8)", "не отвергнута")
    v["verdict_reps"] = np.where(v.rep_lo >= 0.8, "отвергнута (нижняя граница ≥ 0,8)", "не отвергнута")
    v["verdict_both"] = np.where((v.cell_lo >= 0.8) & (v.rep_lo >= 0.8), "отвергнута", np.where((v.cell_lo < 0.8) & (v.rep_lo < 0.8), "не отвергнута", "неопределённо"))
    v[["metric", "n_cells", "tau_b", "cell_lo", "cell_hi", "verdict_cells", "rep_lo", "rep_hi", "verdict_reps", "verdict_both", "degenerate_cells_18"]].to_csv(
        OUT / "t43_h1_rules.csv", index=False, encoding="utf-8")
    print(v[["metric", "tau_b", "cell_lo", "rep_lo", "verdict_cells", "verdict_reps", "verdict_both"]].to_string())

    # ---- В9 методы порога ----
    mrows = []
    for m in meta.itertuples():
        x = by_h[m.config_hash]
        row = {"R": m.R, "Q": m.Q, "D": m.D, "S": m.S}
        vals = {}
        for pr, nm in (("A", "fold"), ("B", "rel")):
            xp = x[x.protocol == pr]
            xx, n, k = aggregate(xp["rho_test"].to_numpy(), xp["failed"].to_numpy().astype(int))
            val, cens, meth = point_threshold(xx, n, k, 0.5, 1.0)
            iso, ic = _isotonic_threshold(xx, n, k, 0.5) if 0 < k.sum() < n.sum() or k.sum() > 0 else (np.nan, "right")
            iso = 1.0 if ic == "right" else iso
            p = k / n
            lin, lc = 1.0, "right"
            if k.sum() == 0:
                lin, lc = 1.0, "right"
            else:
                above = np.flatnonzero(p >= 0.5)
                if len(above) == 0:
                    lin, lc = 1.0, "right"
                elif above[0] == 0:
                    lin, lc = float(xx[0]), "left"
                else:
                    j = int(above[0])
                    lin, lc = float(xx[j - 1] + (0.5 - p[j - 1]) / (p[j] - p[j - 1]) * (xx[j] - xx[j - 1])), "none"
            vals[nm] = (val, cens, meth, iso, ic, lin, lc)
        for nm in ("fold", "rel"):
            v_, c_, m_, iso, ic, lin, lc = vals[nm]
            row.update({f"{nm}_plan": v_, f"{nm}_plan_method": m_, f"{nm}_plan_cens": c_, f"{nm}_iso": iso, f"{nm}_lin": lin, f"{nm}_lin_cens": lc})
        row["delta_plan"] = row["fold_plan"] - row["rel_plan"]
        row["delta_iso"] = row["fold_iso"] - row["rel_iso"]
        row["delta_lin"] = row["fold_lin"] - row["rel_lin"]
        tr = t01[t01.config_hash == m.config_hash].iloc[0]
        row["fold_lo"], row["fold_hi"], row["rel_lo"], row["rel_hi"] = tr.fold_lo, tr.fold_hi, tr.rel_lo, tr.rel_hi
        row["fold_lin_in_ci"] = bool(row["fold_lo"] <= row["fold_lin"] <= row["fold_hi"])
        row["rel_lin_in_ci"] = bool(row["rel_lo"] <= row["rel_lin"] <= row["rel_hi"])
        mrows.append(row)
    mt = pd.DataFrame(mrows)
    mt.to_csv(OUT / "t44_methods.csv", index=False, encoding="utf-8")
    a12 = mt[mt.apply(lambda z: (z.R, z.Q, z.D, z.S) in act, axis=1)]
    print(a12[["R", "D", "S", "fold_plan", "fold_iso", "fold_lin", "fold_lin_in_ci", "rel_plan", "rel_lin", "rel_lin_in_ci", "delta_plan", "delta_lin"]].round(3).to_string())
    eff = []
    for meth in ("plan", "iso", "lin"):
        dl = {(z.R, z.D, z.S): z[f"delta_{meth}"] for _, z in a12.iterrows()}
        d_eff = np.mean([dl[(r, "on", sx)] - dl[(r, "off", sx)] for r in ("R1", "R2") for sx in SS])
        r_eff = np.mean([dl[("R2", d, sx)] - dl[("R1", d, sx)] for d in ("off", "on") for sx in SS])
        rr = {sx: sum(dl[(r, "on", sx)] for r in ("R1", "R2")) / sum(dl[(r, "off", sx)] for r in ("R1", "R2")) for sx in SS}
        eff.append({"method": meth, "D_on_minus_off": d_eff, "R2_minus_R1": r_eff, "r_Exp": rr["Exp"], "r_LogN": rr["LogN"], "r_Pareto": rr["Pareto"],
                    "H1p_diff": rr["Exp"] - rr["LogN"],
                    "max_abs_fold_diff_vs_plan": float((a12[f"fold_{meth}"] - a12["fold_plan"]).abs().max()),
                    "max_abs_rel_diff_vs_plan": float((a12[f"rel_{meth}"] - a12["rel_plan"]).abs().max()),
                    "max_abs_delta_diff_vs_plan": float((a12[f"delta_{meth}"] - a12["delta_plan"]).abs().max())})
    eff = pd.DataFrame(eff)
    eff["n_lin_outside_CI"] = [0, 0, int((~a12.fold_lin_in_ci).sum() + (~a12.rel_lin_in_ci).sum())]
    eff.to_csv(OUT / "t44c_method_effects.csv", index=False, encoding="utf-8")
    print(eff.round(3).to_string())
    h = hmap[("R2", "FIFO", "on", "Exp")]
    xa = by_h[h][by_h[h].protocol == "A"]
    near = xa[(xa.rho_test >= 0.80) & (xa.rho_test <= 0.95)].groupby("rho_test").agg(n=("failed", "size"), P_A=("failed", "mean"), g_mean=("g_final", "mean")).reset_index()
    near.to_csv(OUT / "t44b_r2_on_exp.csv", index=False, encoding="utf-8")
    print(near.round(3).to_string())
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
