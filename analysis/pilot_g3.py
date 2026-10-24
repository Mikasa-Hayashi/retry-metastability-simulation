"""Pilot, сводка для гейта G3: пересчёт ключевых чисел из сырых данных results/pilot/.

python analysis/pilot_g3.py
Не использует seed'ы main (>= 1000000). Вывод: results/pilot/g3_*.csv и results/pilot/g3_summary.txt.
Разделы (номера соответствуют reports/pilot.md):
  S1 полнота и время; S2 опорная ячейка и контроль R0; S3 скрининг 72 конфигураций (повторяет оценки
  pilot4_screen.work по сырым данным и сверяет с pilot4_screen_main.csv); S4 разделимость активных/неактивных ячеек
  при малом числе реплик; S5 H1 (τ-b: Δ и непрерывные метрики); S6 H2 (контрасты Δ и L); S7 H3/H3b (mean-field);
  S8 зависимость от H_rec; S9 мощность (подвыборки из пула 2000 реплик); S10 стоимость основного прогона.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import theory as th  # noqa: E402
from pilot4_common import OUT, cfg_meta, delta_block, load_crn, norm_failed, point_thr  # noqa: E402
from pilot4_screen import work as screen_work  # noqa: E402

from src.runner import check_complete, load_results, stage_dir  # noqa: E402
from src.thresholds import estimate_delta  # noqa: E402

TAGS_MAIN = ["p4_scr", "p4_ext", "p4_ref1", "p4_ref2n"]
NB = 2000
GRID = [round(0.2 + 0.05 * i, 6) for i in range(17)]  # 0,20 … 1,00
LINES: list[str] = []
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


def emit(s: str = "") -> None:
    print(s, flush=True)
    LINES.append(s)


def main_meta() -> pd.DataFrame:
    m = cfg_meta()
    m = m[(m["T"] == 10) & (m.K == 128) & (m.c == 8) & (m.m == 3) & (m.theta == 0.5) & (m.trig == 1.5)
          & (m.h_rec == 1500) & (m.ol_len == 1000) & (m.ol_rho == 1.05) & (m.theta_q == 0.5)]
    return m.sort_values(["R", "Q", "D", "S"]).reset_index(drop=True)


def mf_spinodals(K: int, T: float, theta: float = 0.5) -> tuple[float, float]:
    p = th.MFParams(c=8, k_queue=K, t_out=T)

    def has_h(r: float) -> bool:
        return any(x.stable and th.goodput_ratio(x.q, 3) >= theta for x in th.fixed_points(r, p, 801))

    def has_f(r: float) -> bool:
        return any(x.stable and th.goodput_ratio(x.q, 3) < theta for x in th.fixed_points(r, p, 801))

    fold = th.refine_threshold(has_h, 0.3, 1.2, 1e-3) if (has_h(0.3) and not has_h(1.2)) else float("nan")
    rel = th.refine_threshold(has_f, 0.05, 1.2, 1e-3) if ((not has_f(0.05)) and has_f(1.2)) else float("nan")
    return rel, fold


def tau_boot(x: np.ndarray, y: np.ndarray, nb: int = 2000, seed: int = 7) -> tuple[float, float, float, int]:
    t0 = kendalltau(x, y, variant="b").statistic
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(nb):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0:
            continue
        vals.append(kendalltau(x[i], y[i], variant="b").statistic)
    v = np.array(vals)
    return float(t0), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)


def s1() -> None:
    emit("## S1. Полнота, seed'ы, время")
    rep = check_complete("pilot")
    emit(f"check_complete: {json.dumps(rep, ensure_ascii=False)}")
    r = load_results(stage_dir("pilot"), ["seed", "runtime", "tag", "git_commit", "src_hash", "R", "Q", "protocol", "rho_test"])
    emit(f"строк {len(r)}, seed min {r.seed.min()} max {r.seed.max()} (main начинается с 1000000: пересечения "
         f"{'нет' if r.seed.max() < 1_000_000 else 'ЕСТЬ'})")
    emit(f"src_hash уникальных: {r.src_hash.nunique()} ({sorted(r.src_hash.unique())[:2]}); "
         f"git_commit с суффиксом -dirty: {(r.git_commit.str.endswith('-dirty')).mean():.3f}")
    rt = r.runtime
    emit(f"runtime, с: среднее {rt.mean():.3f}, медиана {rt.median():.3f}, p95 {rt.quantile(.95):.3f}, p99 {rt.quantile(.99):.3f}, "
         f"max {rt.max():.3f}; задач > 1 с: {(rt > 1).sum()} ({(rt > 1).mean():.5f})")
    sess = [json.loads(x) for x in (stage_dir("pilot") / "sessions.jsonl").read_text(encoding="utf-8").splitlines() if x]
    s = pd.DataFrame(sess)
    s["rate"] = s.n_tasks / s.elapsed_s
    p4 = s[s.start >= "2026-10-23 11:30:00"]
    emit(f"сессии: всего {len(s)}, задач {s.n_tasks.sum()}, ошибок {s.n_err.sum()}, время {s.elapsed_s.sum():.0f} с")
    emit(f"сессии с 11:30 23.10 (часть 4; в журнале сессий нет записи о остановленном вручную запуске, 212000 задач без записи): задач {p4.n_tasks.sum()}, время {p4.elapsed_s.sum():.0f} с, "
         f"средняя скорость {p4.n_tasks.sum() / p4.elapsed_s.sum():.1f} задач/с (по сессиям {p4.rate.min():.1f}…{p4.rate.max():.1f})")
    emit(f"исполнителей: {sorted(s.workers.unique())}")


def s2_s3() -> tuple[pd.DataFrame, pd.DataFrame]:
    emit("\n## S2/S3. Скрининг 72 конфигураций (повтор оценки по сырым данным; 40 реплик, ρ ≤ 1 основная, ρ ≤ 2,5 вспомогательная)")
    meta = main_meta()
    emit(f"конфигураций в основной сетке pilot: {len(meta)} (R {meta.R.nunique()}, Q {meta.Q.nunique()}, D {meta.D.nunique()}, S {meta.S.nunique()})")
    cols = ["config_hash", "rho_test", "protocol", "failed", "g_final", "replicate", "wasted_fraction", "amplification",
            "runtime", "recovery_time", "recovery_censored"]
    res = load_crn(cols, TAGS_MAIN)
    res = res[res.config_hash.isin(meta.config_hash) & (res.replicate < 40)].copy()
    res["rho_test"] = res["rho_test"].round(6)
    jobs = [(h, res[res.config_hash == h][["rho_test", "protocol", "failed", "g_final"]], 2.5) for h in meta.config_hash]
    with mp.Pool(max(1, (mp.cpu_count() or 2) - 1)) as pool:
        rows = pool.map(screen_work, jobs, chunksize=2)
    df = meta.merge(pd.DataFrame(rows), on="config_hash")
    df.to_csv(OUT / "g3_screen72.csv", index=False)
    prev = pd.read_csv(OUT / "pilot4_screen_main.csv").set_index("config_hash")
    d = (df.set_index("config_hash")[["le1_delta", "le1_fold", "le1_rel"]] - prev.loc[df.config_hash, ["le1_delta", "le1_fold", "le1_rel"]]).abs()
    emit(f"сверка с pilot4_screen_main.csv: max |разность| Δ {d.le1_delta.max():.2e}, ρ_fold {d.le1_fold.max():.2e}, ρ_release {d.le1_rel.max():.2e}")
    emit(f"N1 (ρ_fold<1 и нижняя граница ДИ Δ>0): {int(df.le1_N1.sum())} из {len(df)}; N2 (Δ≥0,1): {int(df.le1_N2.sum())}")
    emit(f"классы: {df.le1_cls.value_counts().to_dict()}")
    emit(f"цензурировано справа: ρ_fold {int((df.le1_fold_cens == 'right').sum())}, ρ_release {int((df.le1_rel_cens == 'right').sum())}; "
         f"слева: {int((df.le1_rel_cens == 'left').sum()) + int((df.le1_fold_cens == 'left').sum())}")
    for f in ("R", "Q", "D", "S"):
        g = df.groupby(f).agg(n=("config_hash", "size"), N1=("le1_N1", "sum"), N2=("le1_N2", "sum"))
        emit(f"маргинально по {f}: " + "; ".join(f"{i}: N1 {int(r.N1)}/{int(r.n)}, N2 {int(r.N2)}" for i, r in g.iterrows()))
    act = df[df.le1_N1]
    emit("N1-ячейки: " + ", ".join(f"{r.R}/{r.Q}/{r.D}/{r.S}" for r in act.itertuples()))
    emit("\nΔ(ρ ≤ 1) активных ячеек: R Q D S ρ_fold ρ_release Δ [ДИ]")
    for r in act.itertuples():
        emit(f"  {r.R}/{r.Q}/{r.D}/{r.S}: {r.le1_fold:.3f}  {r.le1_rel:.3f}  {r.le1_delta:.3f} [{r.le1_d_lo:.3f}; {r.le1_d_hi:.3f}]")
    r0 = df[df.R == "R0"]
    emit(f"\nКонтроль R0 (18 конфигураций): Δ(ρ≤1) min {r0.le1_delta.min():.3f}, max {r0.le1_delta.max():.3f}; "
         f"нижняя граница ДИ > 0 в {int((r0.le1_d_lo > 0).sum())}; верхняя граница ДИ > 0,02: {int((r0.le1_d_hi > 0.02).sum())}; "
         f"Δ_ext (ρ≤2,5) max {r0.ext_delta.max():.3f}, нижняя граница ДИ > 0: {int((r0.ext_d_lo > 0).sum())}")
    emit("R0 с ДИ Δ(ρ≤1), не содержащим 0 или |Δ|>0,01: " +
         (", ".join(f"{r.Q}/{r.D}/{r.S}={r.le1_delta:.3f}[{r.le1_d_lo:.3f};{r.le1_d_hi:.3f}]" for r in r0.itertuples()
                    if r.le1_d_lo > 0 or r.le1_d_hi < 0 or abs(r.le1_delta) > 0.01) or "нет"))
    emit("R0/ext Δ>0 по ДИ: " + (", ".join(f"{r.Q}/{r.D}/{r.S}={r.ext_delta:.3f}[{r.ext_d_lo:.3f};{r.ext_d_hi:.3f}]" for r in r0.itertuples() if r.ext_d_lo > 0) or "нет"))
    # нормированный критерий
    emit(f"\nНормированный критерий g < θ·min(1,1/ρ), ρ ≤ 2,5 (θ=0,5): ρ_fold цензурировано справа в {int((df.nrm_fold_cens == 'right').sum())} из 72; "
         f"N1 {int(df.nrm_N1.sum())}; N2 {int(df.nrm_N2.sum())}; N1 пересечение с N1(ρ≤1): {int((df.nrm_N1 & df.le1_N1).sum())}")
    emit(f"Простой критерий, ρ ≤ 2,5: ρ_fold цензурировано справа {int((df.ext_fold_cens == 'right').sum())}; N1 {int(df.ext_N1.sum())}; N2 {int(df.ext_N2.sum())}")
    nn = df[~df.le1_N1]
    emit(f"Неактивные 60: простой критерий, порог ρ_fold(ρ≤2,5) медиана {nn.ext_fold.median():.2f}, диапазон [{nn.ext_fold.min():.2f}; {nn.ext_fold.max():.2f}] "
         f"(1/ρ = 0,5 при ρ = 2,0); нормированный: цензурировано справа {int((nn.nrm_fold_cens == 'right').sum())} из {len(nn)}")
    return df, res


def s4(df: pd.DataFrame, res: pd.DataFrame) -> None:
    emit("\n## S4. Разделимость активных и неактивных ячеек при малом числе реплик (ρ ≤ 1, грубая сетка, подвыборка реплик 0…R−1)")
    act = set(df[df.le1_N1].config_hash)
    for nrep in (8, 16, 24, 40):
        sub = res[(res.replicate < nrep) & (res.rho_test <= 1.0 + 1e-9) & (res.rho_test.round(2).isin(GRID))]
        p = sub.groupby(["config_hash", "protocol", "rho_test"]).failed.mean().unstack("protocol")
        gap = (p["B"] - p["A"]).abs().groupby("config_hash").max()
        mx = p.groupby("config_hash").max().max(axis=1)
        a = gap[gap.index.isin(act)]
        n = gap[~gap.index.isin(act)]
        # классификация по точечным порогам
        cls_n2 = []
        for h, g in sub.groupby("config_hash"):
            ra, fa = g[g.protocol == "A"].rho_test.to_numpy(), g[g.protocol == "A"].failed.to_numpy()
            rb, fb = g[g.protocol == "B"].rho_test.to_numpy(), g[g.protocol == "B"].failed.to_numpy()
            va, ca = point_thr(ra, fa, 1.0)
            vb, cb = point_thr(rb, fb, 1.0)
            cls_n2.append((h, ca != "right" and (va - vb) >= 0.1))
        c2 = pd.Series(dict(cls_n2))
        full = df.set_index("config_hash").le1_N2
        agree = int((c2 == full.loc[c2.index]).sum())
        emit(f"R = {nrep}: max_ρ|P_B−P_A|: активные (n={len(a)}) min {a.min():.2f}; неактивные (n={len(n)}) max {n.max():.2f}; "
             f"совпадение точечной N2-классификации с R=40: {agree}/72")
    # неактивные с отказами
    sub = res[(res.rho_test <= 1.0 + 1e-9)]
    pa = sub.groupby(["config_hash", "protocol", "rho_test"]).failed.mean().unstack("protocol")
    nn = pa[~pa.index.get_level_values(0).isin(act)]
    bad = nn[(nn.A > 0) | (nn.B > 0)]
    emit(f"неактивные ячейки, точки ρ ≤ 1 с P>0 (R=40): {len(bad)} из {len(nn)}; max P_A {nn.A.max():.3f}, max P_B {nn.B.max():.3f}")
    byc = bad.groupby(level=0).size()
    mm = df.set_index("config_hash")
    emit("  конфигурации (R/Q/D/S): " + "; ".join(f"{mm.loc[h, 'R']}/{mm.loc[h, 'Q']}/{mm.loc[h, 'D']}/{mm.loc[h, 'S']}: {n} точек" for h, n in byc.items()))


def metric_arrays(res: pd.DataFrame, df: pd.DataFrame):
    """g[config] -> {A,B}: [n_rho, n_rep] на сетке GRID (ρ ≤ 1)."""
    out = {}
    for h in df.config_hash:
        d = res[(res.config_hash == h) & res.rho_test.isin(GRID)]
        out[h] = {pr: d[d.protocol == pr].pivot_table(index="rho_test", columns="replicate", values="g_final", aggfunc="first")
                  .reindex(index=GRID, columns=range(40)).to_numpy() for pr in ("A", "B")}
    return out


def s5_s6(df: pd.DataFrame, res: pd.DataFrame) -> None:
    emit("\n## S5. H1: коэффициент Кендалла τ-b между рейтингами при Exp и LogN (bootstrap по ячейкам, 2000)")
    arr = metric_arrays(res, df)
    grid = np.array(GRID)
    gstar = np.minimum(1.0, 1.0 / grid)
    rows = []
    for r in df.itertuples():
        a, b = arr[r.config_hash]["A"], arr[r.config_hash]["B"]
        la_rep = np.trapezoid(gstar[:, None] - a, grid, axis=0)
        lb_rep = np.trapezoid(gstar[:, None] - b, grid, axis=0)
        h_ = np.trapezoid(np.abs(np.nanmean(b, 1) - np.nanmean(a, 1)), grid)
        rows.append({"config_hash": r.config_hash, "R": r.R, "Q": r.Q, "D": r.D, "S": r.S, "delta": r.le1_delta,
                     "LA": float(np.nanmean(la_rep)), "LB": float(np.nanmean(lb_rep)), "H": float(h_), "LA_rep": la_rep, "LB_rep": lb_rep})
    mt = pd.DataFrame(rows)
    mt.drop(columns=["LA_rep", "LB_rep"]).to_csv(OUT / "g3_metrics72.csv", index=False)
    for nm, sel in (("18 ячеек без R0", mt.R != "R0"), ("24 ячейки с R0", np.ones(len(mt), bool))):
        sub = mt[sel]
        for metric in ("delta", "LA", "H"):
            pv = sub.pivot_table(index=["R", "Q", "D"], columns="S", values=metric)
            t, lo, hi, nv = tau_boot(pv["Exp"].to_numpy(), pv["LogN"].to_numpy())
            ties = np.mean([x == y for i, x in enumerate(pv["Exp"].round(2)) for y in pv["Exp"].round(2).iloc[i + 1:]])
            emit(f"  {nm}, метрика {metric}: τ-b(Exp, LogN) = {t:.3f}, ДИ по ячейкам [{lo:.3f}; {hi:.3f}] (валидных {nv}/2000); доля связок в Exp после округления до 0,01: {ties:.2f}")
    # активные
    act = mt[(mt.R.isin(["R1", "R2"])) & (mt.Q == "FIFO")]
    pv = act.pivot_table(index=["R", "D"], columns="S", values="delta")
    emit("  4 активные ячейки (R1/R2 × FIFO × D): Δ при Exp/LogN/Pareto:")
    for i, r in pv.iterrows():
        emit(f"    {i[0]}/FIFO/{i[1]}: {r['Exp']:.3f} / {r['LogN']:.3f} / {r['Pareto']:.3f}")
    t, lo, hi, nv = tau_boot(pv["Exp"].to_numpy(), pv["LogN"].to_numpy())
    emit(f"  τ-b по 4 активным ячейкам (Exp, LogN) = {t:.3f} [{lo:.3f}; {hi:.3f}] (валидных {nv})")
    # относительное снижение Δ от D=on
    emit("  Относительное остаточное Δ при D=on / D=off (активные, усреднено по R1,R2):")
    for s in ("Exp", "LogN", "Pareto"):
        on = act[(act.S == s) & (act.D == "on")].delta.mean()
        off = act[(act.S == s) & (act.D == "off")].delta.mean()
        emit(f"    {s}: Δ_off {off:.3f}, Δ_on {on:.3f}, отношение {on / off:.3f}")
    emit("  R2 − R1 по Δ (FIFO): " + "; ".join(
        f"{d}/{s}: {act[(act.R == 'R2') & (act.D == d) & (act.S == s)].delta.iloc[0] - act[(act.R == 'R1') & (act.D == d) & (act.S == s)].delta.iloc[0]:+.3f}"
        for d in ("off", "on") for s in ("Exp", "LogN", "Pareto")))

    emit("\n## S6. H2: контраст R3×ALIFO (ρ ≤ 1)")
    emit("  C(Δ) = Δ(R3,ALIFO) − Δ(R3,FIFO) − Δ(R2,ALIFO) + Δ(R2,FIFO) по (D, S):")
    cs = []
    for d in ("off", "on"):
        for s in ("Exp", "LogN", "Pareto"):
            g = lambda r, q, d=d, s=s: float(mt[(mt.R == r) & (mt.Q == q) & (mt.D == d) & (mt.S == s)].delta.iloc[0])  # noqa: E731
            c_ = g("R3", "ALIFO") - g("R3", "FIFO") - g("R2", "ALIFO") + g("R2", "FIFO")
            cs.append(c_)
            emit(f"    D={d}, S={s}: Δ(R3,ALIFO) {g('R3', 'ALIFO'):+.3f}, Δ(R3,FIFO) {g('R3', 'FIFO'):+.3f}, Δ(R2,ALIFO) {g('R2', 'ALIFO'):+.3f}, "
                 f"Δ(R2,FIFO) {g('R2', 'FIFO'):+.3f}; C = {c_:+.3f}")
    emit(f"  C<0 в {sum(c_ < 0 for c_ in cs)} из 6 ячеек; min {min(cs):.3f}, max {max(cs):.3f}")
    # непрерывные метрики с bootstrap по репликам (общие индексы)
    rng = np.random.default_rng(np.random.SeedSequence(2027))
    idx = rng.integers(0, 40, (NB, 40))
    lab = mt[["R", "Q", "D", "S"]]

    def cell_boot(rep_col: str, r: str, q: str) -> np.ndarray:
        m_ = ((lab.R == r) & (lab.Q == q)).to_numpy()
        arrs = np.stack([mt.loc[m_, rep_col].iloc[i] for i in range(m_.sum())])  # [cells, 40]
        return np.stack([arrs[:, ix].mean(axis=(0, 1)) if False else arrs[:, ix].mean(axis=1).mean(axis=0) for ix in idx])  # [NB]

    emit("  L_A(ρ≤1) (потерянный goodput, протокол A), усреднено по D и S; bootstrap по репликам (общие индексы), 2000:")
    for col, nm in (("LA_rep", "L_A"), ("LB_rep", "L_B")):
        cb = {(r, q): cell_boot(col, r, q) for r in ("R2", "R3") for q in ("FIFO", "ALIFO")}
        cp = {k: float(np.mean([np.mean(x) for x in mt.loc[((lab.R == k[0]) & (lab.Q == k[1])).to_numpy(), col]])) for k in cb}
        for nm2, fn in (("C(R3×ALIFO)", lambda f: f(("R3", "ALIFO")) - f(("R3", "FIFO")) - f(("R2", "ALIFO")) + f(("R2", "FIFO"))),
                        ("R3−R2 | ALIFO", lambda f: f(("R3", "ALIFO")) - f(("R2", "ALIFO"))),
                        ("R3−R2 | FIFO", lambda f: f(("R3", "FIFO")) - f(("R2", "FIFO"))),
                        ("ALIFO−FIFO | R2", lambda f: f(("R2", "ALIFO")) - f(("R2", "FIFO")))):
            est = fn(lambda k, cp=cp: cp[k])
            bb = fn(lambda k, cb=cb: cb[k])
            lo, hi = np.percentile(bb, [2.5, 97.5])
            emit(f"    {nm} {nm2}: {est:+.4f} [{lo:+.4f}; {hi:+.4f}]")
    # уровни L_A
    emit("  L_A(ρ≤1), L_B(ρ≤1) для S=Exp, D=off:")
    for r, q in (("R0", "FIFO"), ("R1", "FIFO"), ("R2", "FIFO"), ("R3", "FIFO"), ("R1", "LIFO"), ("R1", "ALIFO"), ("R2", "ALIFO"), ("R3", "ALIFO")):
        x = mt[(mt.R == r) & (mt.Q == q) & (mt.D == "off") & (mt.S == "Exp")].iloc[0]
        emit(f"    {r}/{q}: L_A {x.LA:.4f}, L_B {x.LB:.4f}, H {x.H:.4f}")
    # wasted/amplification при ρ=0.9 A
    w = res[(res.protocol == "A") & (res.rho_test.round(2) == 0.9)].merge(main_meta()[["config_hash", "R", "Q", "D", "S"]], on="config_hash")
    wm = w.groupby(["R", "Q", "D", "S"])[["wasted_fraction", "amplification"]].mean()
    emit("  ρ=0,9, протокол A, S=Exp, D=off: wasted / усиление:")
    for r, q in (("R0", "FIFO"), ("R1", "FIFO"), ("R3", "FIFO"), ("R1", "LIFO"), ("R1", "ALIFO"), ("R2", "ALIFO"), ("R3", "ALIFO")):
        x = wm.loc[(r, q, "off", "Exp")]
        emit(f"    {r}/{q}: {x.wasted_fraction:.3f} / {x.amplification:.3f}")
    mt.drop(columns=["LA_rep", "LB_rep"]).to_csv(OUT / "g3_metrics72.csv", index=False)


def ref_hash(meta: pd.DataFrame, r: str = "R1") -> str:
    return str(meta[(meta.R == r) & (meta.Q == "FIFO") & (meta.D == "off") & (meta.S == "Exp")].config_hash.iloc[0])


def s2b(res: pd.DataFrame) -> None:
    emit("\n## S2b. P_A(ρ), P_B(ρ) опорной ячейки (R1/FIFO/off/Exp) и R0/FIFO/off/Exp, 40 реплик")
    meta = main_meta()
    for r in ("R1", "R0"):
        h = ref_hash(meta, r)
        d = res[(res.config_hash == h) & res.rho_test.isin(GRID)]
        p = d.groupby(["protocol", "rho_test"]).failed.mean().unstack("protocol")
        n = d.groupby(["protocol", "rho_test"]).size().unstack("protocol")
        emit(f"{r} (hash {h}; реплик в точке A {n.A.min()}…{n.A.max()}, B {n.B.min()}…{n.B.max()}):")
        emit("  ρ: " + " ".join(f"{x:.2f}" for x in p.index))
        emit("  P_A: " + " ".join(f"{x:.3f}" for x in p.A))
        emit("  P_B: " + " ".join(f"{x:.3f}" for x in p.B))
        pos = p[p.A > 0]
        emit(f"  минимальное ρ ≤ 1 с P_A > 0: {pos.index.min() if len(pos) else 'нет'}; минимальное с P_A ≥ 0,5: {p[p.A >= .5].index.min() if (p.A >= .5).any() else 'нет'}; "
             f"P_B ≥ 0,5 начиная с ρ ≥: {p[p.B < .5].index.max() if (p.B < .5).any() else 'нет'} (последнее ρ с P_B < 0,5)")


def s7(meta: pd.DataFrame) -> None:
    emit("\n## S7. H3 / H3b: mean-field (analysis/theory.py, R1, FIFO, Exp, m=3) против симуляции")
    m = cfg_meta()
    ref = m[(m.R == "R1") & (m.Q == "FIFO") & (m.D == "off") & (m.S == "Exp") & (m.theta_q == 0.5) & (m.m == 3) & (m.c == 8)
            & (m.h_rec == 1500) & (m.ol_len == 1000) & (m.theta == 0.5)]
    tags = TAGS_MAIN + ["p4_sens", "p4_sens_ref"]
    res = load_crn(["config_hash", "rho_test", "protocol", "failed", "replicate"], tags)
    res = res[res.replicate < 40]
    rows = []
    for T, K, trig in ((10, 128, 1.5), (5, 128, 1.5), (10, 256, 1.5), (10, 128, 1.25), (10, 128, 1.75), (20, 128, 1.5), (10, 64, 1.5)):
        q = ref[(ref["T"] == T) & (ref.K == K) & (ref.trig == trig)]
        if q.empty:
            continue
        h = q.config_hash.iloc[0]
        hb = ref[(ref["T"] == T) & (ref.K == K) & (ref.trig == 1.5)].config_hash.iloc[0]  # B не зависит от триггера (D-015)
        a = res[(res.config_hash == h) & (res.protocol == "A")]
        b = res[(res.config_hash == hb) & (res.protocol == "B")]
        hi = 1.5
        e = estimate_delta(a.rho_test.to_numpy(), a.failed.to_numpy(), b.rho_test.to_numpy(), b.failed.to_numpy(), NB, 11, right_bound=hi)
        rel_mf, fold_mf = mf_spinodals(K, float(T))
        rows.append({"T": T, "K": K, "m_s": trig, "fold": e.fold.value, "fold_c": e.fold.censored, "rel": e.release.value, "rel_c": e.release.censored,
                     "delta": e.delta, "d_lo": e.lo, "d_hi": e.hi, "mf_rel": rel_mf, "mf_fold": fold_mf,
                     "rel_minus_mf": e.release.value - rel_mf, "rel_lo_minus": e.release.lo - rel_mf, "rel_hi_minus": e.release.hi - rel_mf,
                     "fold_minus_mf": e.fold.value - fold_mf, "fold_lo_minus": e.fold.lo - fold_mf, "fold_hi_minus": e.fold.hi - fold_mf})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "g3_h3.csv", index=False)
    for r in t.itertuples():
        emit(f"  T={r.T}, K={r.K}, m_s={r.m_s}: sim ρ_fold {r.fold:.3f} ({r.fold_c}), ρ_release {r.rel:.3f} ({r.rel_c}), Δ {r.delta:.3f} [{r.d_lo:.3f}; {r.d_hi:.3f}]; "
             f"MF ρ_fold {r.mf_fold:.3f}, ρ_release {r.mf_rel:.3f}; sim−MF: ρ_release {r.rel_minus_mf:+.3f} [{r.rel_lo_minus:+.3f}; {r.rel_hi_minus:+.3f}], "
             f"ρ_fold {r.fold_minus_mf:+.3f} [{r.fold_lo_minus:+.3f}; {r.fold_hi_minus:+.3f}]")
    # LogN: теория неприменима; сравнение с MF(Exp) только для справки
    mm = main_meta()
    full = pd.read_csv(OUT / "g3_screen72.csv")
    for s in ("LogN", "Pareto"):
        x = full[(full.R == "R1") & (full.Q == "FIFO") & (full.D == "off") & (full.S == s)].iloc[0]
        emit(f"  Справка (MF неприменима к {s}): sim ρ_release {x.le1_rel:.3f}, ρ_fold {x.le1_fold:.3f}; MF(Exp) ρ_release 0,277: разность {x.le1_rel - 0.277:+.3f}")
    del mm


def s8() -> None:
    emit("\n## S8. Зависимость от H_rec (часть 1, перегрузка B = 400, триггер (1,5; 25), опорная ячейка, 40 реплик)")
    r = load_results(stage_dir("pilot"), ["tag", "config_hash", "rho_test", "protocol", "failed", "replicate", "trigger_mult", "trigger_len"])
    r = r[r.replicate < 40]
    sets = {1500: ("calib_A", "refine_A", "calib_B", "refine_B"), 3000: ("hrec3000_A", "hrec3000_B"), 6000: ("hrec6000_A", "hrec6000_B")}
    for h, tg in sets.items():
        d = r[r.tag.isin(tg)]
        a = d[(d.protocol == "A") & (d.trigger_mult == 1.5) & (d.trigger_len == 25)]
        b = d[d.protocol == "B"]
        e = estimate_delta(a.rho_test.to_numpy(), a.failed.to_numpy(), b.rho_test.to_numpy(), b.failed.to_numpy(), NB, 11, right_bound=1.0)
        emit(f"  H_rec={h}: sim − MF(ρ_release 0,277) = {e.release.value - 0.277:+.3f} [{e.release.lo - 0.277:+.3f}; {e.release.hi - 0.277:+.3f}]")
        emit(f"  H_rec={h}: ρ_fold {e.fold.value:.3f} [{e.fold.lo:.3f}; {e.fold.hi:.3f}], ρ_release {e.release.value:.3f} [{e.release.lo:.3f}; {e.release.hi:.3f}], "
             f"Δ {e.delta:.3f} [{e.lo:.3f}; {e.hi:.3f}]")


def s9() -> None:
    emit("\n## S9. Мощность: полуширина bootstrap-ДИ по подвыборкам из независимого пула (опорная ячейка, окна ρ_fold 0,80…0,90, ρ_release 0,29…0,39, шаг 0,01)")
    r = load_results(stage_dir("pilot"), ["tag", "config_hash", "rho_test", "protocol", "failed", "replicate", "seed"])
    pa = r[r.tag == "p4_cov_P_A"]
    pb = r[r.tag == "p4_cov_P_B"]
    nrep = min(pa.groupby(pa.rho_test.round(4)).size().min(), pb.groupby(pb.rho_test.round(4)).size().min())
    emit(f"пул: A {len(pa)} строк, B {len(pb)} строк, реплик в точке ≥ {nrep}; seed'ы {int(pa.seed.min())}…{int(pb.seed.max())}")
    ra, rb = pa.sort_values("seed"), pb.sort_values("seed")
    rng = np.random.default_rng(np.random.SeedSequence(99))
    rows = []
    for R in (16, 24, 40, 60, 100):
        hw_f, hw_r, hw_d, est_f, est_r, est_d = [], [], [], [], [], []
        for _ in range(30):
            def pick(d: pd.DataFrame) -> pd.DataFrame:
                out = []
                for rho, g in d.groupby(d.rho_test.round(4)):
                    out.append(g.iloc[rng.choice(len(g), R, replace=False)])
                return pd.concat(out)
            a, b = pick(ra), pick(rb)
            e = estimate_delta(a.rho_test.to_numpy(), a.failed.to_numpy(), b.rho_test.to_numpy(), b.failed.to_numpy(), 500, int(rng.integers(1e9)), right_bound=1.0)
            hw_f.append((e.fold.hi - e.fold.lo) / 2)
            hw_r.append((e.release.hi - e.release.lo) / 2)
            hw_d.append((e.hi - e.lo) / 2)
            est_f.append(e.fold.value)
            est_r.append(e.release.value)
            est_d.append(e.delta)
        rows.append({"R": R, "hw_fold": np.mean(hw_f), "hw_rel": np.mean(hw_r), "hw_delta": np.mean(hw_d),
                     "sd_fold": np.std(est_f, ddof=1), "sd_rel": np.std(est_r, ddof=1), "sd_delta": np.std(est_d, ddof=1)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "g3_power.csv", index=False)
    for x in t.itertuples():
        emit(f"  R={x.R}: полуширина ДИ ρ_fold {x.hw_fold:.4f}, ρ_release {x.hw_rel:.4f}, Δ {x.hw_delta:.4f}; SD оценки между подвыборками (30) Δ {x.sd_delta:.4f}")
    emit("  (подвыборки без возвращения из 2000 реплик; bootstrap 500 вместо 2000 для скорости; окно из 11 точек на протокол — полуширина занижена относительно полной сетки не более чем на ширину окна)")


def s10(df: pd.DataFrame) -> None:
    emit("\n## S10. Стоимость основного прогона (runtime по данным части 4, блок p4_scr/p4_ext/p4_ref*)")
    meta = main_meta()
    r = load_crn(["config_hash", "rho_test", "protocol", "runtime", "replicate"], TAGS_MAIN)
    r = r[r.config_hash.isin(meta.config_hash)].merge(df[["config_hash", "Q", "le1_N1"]], on="config_hash")
    r["hi"] = r.rho_test > 1.0 + 1e-9
    g = r.groupby(["le1_N1", "protocol", "hi"]).runtime.agg(["mean", "size"])
    for k, v in g.iterrows():
        emit(f"  активная={k[0]}, протокол {k[1]}, ρ>1={k[2]}: среднее {v['mean']:.3f} с, задач {int(v['size'])}")
    sess = [json.loads(x) for x in (stage_dir("pilot") / "sessions.jsonl").read_text(encoding="utf-8").splitlines() if x]
    s = pd.DataFrame(sess)
    s = s[s.start >= "2026-10-23 11:30:00"]
    allr = load_results(stage_dir("pilot"), ["seed", "runtime"])
    p4rt = allr[allr.seed >= 100_000].runtime.sum()
    eff = 1.0  # сессии покрывают не все задачи части 4 (см. S1), эффективность не оценивается
    emit(f"  часть 4: средний runtime {p4rt / (allr.seed >= 100_000).sum():.3f} с на задачу (по всем {(allr.seed >= 100_000).sum()} задачам); "
         f"скорость по журналу сессий {s.n_tasks.sum() / s.elapsed_s.sum():.1f} задач/с, 7 процессов")
    # меню плана
    a_lo = r[(r.le1_N1) & (~r.hi)].runtime.mean()
    n_lo = r[(~r.le1_N1) & (~r.hi)].runtime.mean()
    n_hi = r[(~r.le1_N1) & (r.hi)].runtime.mean()
    a_hi = r[(r.le1_N1) & (r.hi)].runtime.mean()
    tasks_act = 12 * (17 + 8) * 2 * 100
    tasks_non = 60 * 17 * 2 * 40
    tasks_aux = 72 * 16 * 2 * 16
    tasks_sens = 12 * (17 + 8) * 2 * 40 * 7
    t_act = tasks_act * a_lo
    t_non = tasks_non * n_lo
    t_aux = tasks_aux * (0.5 * n_hi + 0.5 * a_hi)
    t_sens = tasks_sens * a_lo
    emit(f"  вариант плана: активные 12×(17+≈8)×2×100 = {tasks_act}; неактивные 60×17×2×40 = {tasks_non}; ρ>1 вспомогательная 72×16×2×16 = {tasks_aux}; "
         f"чувствительность 12×25×2×40×7 = {tasks_sens}")
    tot_tasks = tasks_act + tasks_non + tasks_aux + tasks_sens
    tot_cpu = t_act + t_non + t_aux + t_sens
    emit(f"  итого задач {tot_tasks}; CPU-секунд {tot_cpu:.0f}; на 7 процессах (runtime части 4, без поправки): {tot_cpu / (7 * eff) / 3600:.2f} ч; по скорости сессий части 4: "
         f"{tot_tasks / (s.n_tasks.sum() / s.elapsed_s.sum()) / 3600:.2f} ч")
    emit(f"  подзадачи, ч (по скорости сессий): активные {tasks_act / (s.n_tasks.sum() / s.elapsed_s.sum()) / 3600:.2f}; неактивные {tasks_non / (s.n_tasks.sum() / s.elapsed_s.sum()) / 3600:.2f}; "
         f"ρ>1 {tasks_aux / (s.n_tasks.sum() / s.elapsed_s.sum()) / 3600:.2f}; чувствительность {tasks_sens / (s.n_tasks.sum() / s.elapsed_s.sum()) / 3600:.2f}")


def main() -> int:
    s1()
    df, res = s2_s3()
    s2b(res)
    s4(df, res)
    s5_s6(df, res)
    s7(main_meta())
    s8()
    s9()
    s10(df)
    (OUT / "g3_summary.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
