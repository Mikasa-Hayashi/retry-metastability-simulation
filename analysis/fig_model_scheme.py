"""Схема модели (рис. 1 статьи): источник, клиент с таймаутом и повторами, очередь, воркеры.

Вертикальная компоновка: при ширине A5 (108 мм) подписи остаются читаемыми.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(5.8, 6.6))
ax.set_xlim(0, 14)
ax.set_ylim(-1.3, 15.5)
ax.axis("off")
C_BOX, C_Q, C_W, C_RET = "#eef2f7", "#fff4d6", "#dff0e3", "#fbe3e0"
RED, GREEN = "#a3392f", "#2d6a3f"
FS = 9


def box(x, y, w, h, text, fc, ec="#333", fs=FS, color="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.2", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color)


def arrow(p, q, color="#333", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=1.4, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


X0, W = 0.3, 7.0
XC = X0 + W / 2
box(X0, 13.0, W, 2.2, "Источник первичных запросов:\nпоток Пуассона, интенсивность λ,\nρ = λE[S]/c", C_BOX)
box(X0, 9.3, W, 2.2, "Клиент: таймаут T = 10,\nполитика повторов R\n(R0–R3, не более 3 повторов)", C_BOX)
arrow((XC, 13.0), (XC, 11.5))
# очередь: подпись внутри, ячейки в нижней части
ax.add_patch(Rectangle((X0, 5.8), W, 2.4, fc=C_Q, ec="#333", lw=1.2))
for i in range(1, 10):
    ax.plot([X0 + i * W / 10] * 2, [5.8, 6.8], color="#999", lw=0.8)
ax.plot([X0, X0 + W], [6.8, 6.8], color="#999", lw=0.8)
ax.text(XC, 7.5, "Очередь ёмкости K = 128, дисциплина Q:\nFIFO, LIFO, ALIFO", ha="center", va="center", fontsize=FS - 1.2)
arrow((XC, 9.3), (XC, 8.2))
ax.text(XC + 0.3, 8.75, "попытка", ha="left", va="center", fontsize=FS - 1, style="italic")
# воркеры
box(X0, 0.8, W, 4.0, "", C_W)
ax.text(XC, 4.4, "c = 8 воркеров", ha="center", va="center", fontsize=FS)
for k in range(4):
    ax.add_patch(Rectangle((X0 + 0.55 + k * 1.55, 3.0), 1.1, 0.8, fc="white", ec="#333", lw=0.9))
    ax.add_patch(Rectangle((X0 + 0.55 + k * 1.55, 2.0), 1.1, 0.8, fc="white", ec="#333", lw=0.9))
ax.text(XC, 1.35, "время обслуживания S, E[S] = 1:\nExp, LogN (cv = 2), Pareto (α = 2,5)", ha="center", va="center", fontsize=FS - 1)
arrow((XC, 5.8), (XC, 4.8))
# успех
arrow((X0 + W, 2.8), (9.0, 2.8), color=GREEN)
ax.text(9.2, 2.8, "успех: ответ\nполучен за\nвремя ≤ T", ha="left", va="center", fontsize=FS - 0.5, color=GREEN)
# отказ / таймаут -> повтор
box(8.7, 9.0, 5.0, 3.4, "отказ или таймаут →\nповтор через паузу\nпо политике R\n(нагрузка от повторов\nрастёт)", C_RET, ec=RED, fs=FS - 0.5, color=RED)
arrow((X0 + W, 7.0), (9.4, 9.0), color=RED, rad=-0.25)
ax.text(8.9, 7.2, "очередь полна:\nотказ", ha="left", va="center", fontsize=FS - 1, color=RED)
arrow((8.7, 10.7), (X0 + W, 10.7), color=RED)
ax.text(0.3, -0.2, "D = on: воркер пропускает запись с истёкшим сроком;\nD = off: обслуживает её впустую.",
        ha="left", va="top", fontsize=FS - 1.5, style="italic")
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"figures/F00_skhema_modeli.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
