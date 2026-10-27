"""Схема модели (рис. 1 статьи): источник, клиент с таймаутом и повторами, очередь, воркеры."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(9.5, 4.0))
ax.set_xlim(0, 19)
ax.set_ylim(0, 7.6)
ax.axis("off")
C_BOX, C_Q, C_W, C_RET = "#eef2f7", "#fff4d6", "#dff0e3", "#fbe3e0"
RED, GREEN = "#a3392f", "#2d6a3f"


def box(x, y, w, h, text, fc, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.2", fc=fc, ec="#333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(p, q, color="#333", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=1.3, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


box(0.2, 3.4, 3.9, 1.8, "Источник первичных\nзапросов: поток Пуассона,\nинтенсивность λ,\nρ = λE[S]/c", C_BOX)
box(5.0, 3.4, 3.9, 1.8, "Клиент: таймаут T = 10,\nполитика повторов R\n(R0–R3, не более\n3 повторов)", C_BOX)
arrow((4.1, 4.3), (5.0, 4.3))
ax.add_patch(Rectangle((10.0, 3.6), 3.8, 1.4, fc=C_Q, ec="#333", lw=1.2))
for i in range(1, 8):
    ax.plot([10.0 + i * 3.8 / 8] * 2, [3.6, 5.0], color="#999", lw=0.8)
ax.text(11.9, 5.55, "Очередь ёмкости K = 128,\nдисциплина Q: FIFO, LIFO, ALIFO", ha="center", fontsize=8.5)
arrow((8.9, 4.3), (10.0, 4.3))
ax.text(9.45, 4.8, "попытка", ha="center", fontsize=7.5)
ax.add_patch(FancyBboxPatch((14.8, 3.0), 3.7, 2.6, boxstyle="round,pad=0.05,rounding_size=0.2", fc=C_W, ec="#333", lw=1.2))
ax.text(16.65, 5.2, "c = 8 воркеров", ha="center", fontsize=8.5)
for k in range(4):
    ax.add_patch(Rectangle((15.1 + k * 0.82, 3.8), 0.62, 0.9, fc="white", ec="#333", lw=0.9))
    ax.add_patch(Rectangle((15.1 + k * 0.82, 3.15), 0.62, 0.5, fc="white", ec="#333", lw=0.9))
arrow((13.8, 4.3), (14.8, 4.3))
ax.text(16.65, 2.45, "время обслуживания S, E[S] = 1:\nExp, LogN (cv = 2), Pareto (α = 2,5)", ha="center", va="center", fontsize=7.5)
arrow((16.65, 5.65), (16.65, 6.6), color=GREEN)
ax.text(16.65, 7.1, "успех: ответ получен за время ≤ T", ha="center", fontsize=8, color=GREEN)
# отказ: очередь полна -> блок повтора -> клиент
arrow((12.4, 3.6), (12.4, 2.0), color=RED)
ax.text(12.5, 2.8, "очередь полна: отказ", ha="left", fontsize=7.5, color=RED)
ax.add_patch(FancyBboxPatch((5.0, 0.9), 7.6, 1.1, boxstyle="round,pad=0.05,rounding_size=0.2", fc=C_RET, ec=RED, lw=1.0))
ax.text(8.8, 1.45, "отказ или таймаут → повтор через паузу по политике R\n(нагрузка от повторов растёт)", ha="center", va="center", fontsize=8, color=RED)
arrow((7.0, 2.0), (7.0, 3.4), color=RED)
ax.text(0.2, 0.25, "D = on: воркер пропускает запись с истёкшим сроком; D = off: обслуживает её впустую.",
        ha="left", fontsize=7.5, style="italic")
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"figures/F00_skhema_modeli.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
