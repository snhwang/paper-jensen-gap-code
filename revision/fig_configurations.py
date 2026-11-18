"""Figure 2: what the gaps see in six tissue configurations, in 3D, at two b-values. Theory only.

The rendering follows the fibre-case talk figures of paper-jensen-gap-info-theory
(analysis/build_fiber_schematics.py), which no manuscript used.

Each voxel holds Gaussian tissue components. The fibres have D_par 1.7 and D_perp 0.3e-3 mm^2/s (FA 0.80).
(a) One fibre along z. (b) A restricted fibre along z: an intra-axonal stick (fraction 0.6, D_par 1.7, D_perp 0)
and an extra-axonal compartment (fraction 0.4, D_par 1.7, D_perp 0.75e-3 mm^2/s). At low b its tensor is that of
(a), so it differs from (a) only by non-Gaussian diffusion. (c) A dispersed bundle: fibres on 2,000 evenly spread
axes (every 20th of the 40,000 directions), weighted by a Watson distribution about z with concentration 4 (order
parameter 0.56). (d) Two fibres along x and y, a 90 degree crossing. (e) The same 2,000 axes with equal weights,
random orientations. (f) An isotropic component with the MD of (a). The voxel signal along each direction is the
weighted mean of the component signals.

Rows: the component diffusion ellipsoids (semi-axes proportional to the square root of the eigenvalues,
superimposed, 24 drawn for (c) and (e)); the profile of the tensor fitted to all directions at b = 1500; the
directional ADC D(g) = -ln S(g)/b at b = 1500 and at b = 3000 s/mm^2. In each column the surfaces share one radial
scale, the maximum of the b = 1500 profile, and colour gives the directional diffusivity (D(g), or g^T D g for the
fitted tensor) on a common scale. These surfaces are unshaded and translucent. For depth they carry lines of latitude and
longitude every 45 degrees and a small x, y, z triad at the origin, and the colorbar shows magma as they show it on white. Bottom: FA of the fitted tensor, the
diffusivity gap with that of the fitted tensor, the signal gap (all at b = 1500), and the shell difference
Delta_b J_ln = J_ln(3000) - J_ln(1500), with its percentage of J_ln(1500). All values from 40,000 near-uniform
directions, as in theory_table.py.

Delta_b J_ln is zero for one Gaussian fibre and for an isotropic voxel. A mixture of compartments or of
orientations makes the voxel non-Gaussian. Delta_b J_ln responds only where that depends on direction, and its sign
shows where the fractional fall of the directional ADC with b is larger. It is positive for the restricted fibre (larger fall
across the fibre), negative for the dispersed bundle (larger fall along it), and zero for random orientations,
although their ADC falls with b.

The output text also records, for Section 2.3, the signal gap of one component, the multidimensional-encoding contrast
ln S_SM(linear) - ln S(isotropic) (equal to it when the components are alike; blank otherwise), the ratio of the voxel's signal gap to
the component's, and its second-order limit V_lambda(mean tensor) / V_lambda(component), which is the square of the
order parameter of Lasic et al. (2014).

Output:  figures/fig_configurations.pdf (.png), out/fig_configurations.txt
Usage:   python fig_configurations.py
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from config import FIGS, OUT

matplotlib.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "pdf.fonttype": 42})
INK, MUTED = "0.15", "0.55"
COLORS = ["#c0392b", "#1f5fa8"]
B1, B2 = 1500.0, 3000.0
E3 = 1e-3
STRONG = (1.7 * E3, 0.3 * E3)
MD_STRONG = (STRONG[0] + 2 * STRONG[1]) / 3
KAPPA = 4.0

N = 40000
k = np.arange(N) + 0.5
ph, th = np.arccos(1 - 2 * k / N), np.pi * (1 + 5 ** 0.5) * k
G = np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)
X = np.c_[G[:, 0] ** 2, G[:, 1] ** 2, G[:, 2] ** 2, 2 * G[:, 0] * G[:, 1], 2 * G[:, 0] * G[:, 2], 2 * G[:, 1] * G[:, 2]]

# grid for drawing surfaces
TH, PH = np.meshgrid(np.linspace(0, np.pi, 41), np.linspace(0, 2 * np.pi, 81), indexing="ij")
GS = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1)


def component(u, shape=STRONG):
    u = np.asarray(u, float) / np.linalg.norm(u)
    return shape[1] * np.eye(3) + (shape[0] - shape[1]) * np.outer(u, u)


def proj(g, T):
    return np.einsum("...i,ij,...j->...", g, T, g)


def signal(g, comps, w, b):
    return sum(wk * np.exp(-b * proj(g, T)) for T, wk in zip(comps, w))


def fa_of(T):
    lam = np.linalg.eigvalsh(T)
    return np.sqrt(1.5 * ((lam - lam.mean()) ** 2).sum() / (lam ** 2).sum())


def v_lambda(T):
    lam = np.linalg.eigvalsh(T)
    return ((lam - lam.mean()) ** 2).mean()


def gap(x):
    return np.log(x.mean()) - np.log(x).mean()


def fit_tensor(D):
    c = np.linalg.lstsq(X, D, rcond=None)[0]
    return np.array([[c[0], c[3], c[4]], [c[3], c[1], c[5]], [c[4], c[5], c[2]]])


AXES = G[::20]                                                   # 2,000 evenly spread fibre axes
W_DISP = np.exp(KAPPA * AXES[:, 2] ** 2); W_DISP /= W_DISP.sum()
W_RAND = np.full(len(AXES), 1 / len(AXES))
FIBRES = [component(u) for u in AXES]
F_INTRA, D_EXTRA_PERP = 0.6, 0.75 * E3                          # low-b tensor equals the fibre of (a)
RESTRICTED = [component([0, 0, 1], (STRONG[0], 0.0)), component([0, 0, 1], (STRONG[0], D_EXTRA_PERP))]
CONFIGS = [("(a) Coherent", [component([0, 0, 1])], [1.0]),
           ("(b) Restricted", RESTRICTED, [F_INTRA, 1 - F_INTRA]),
           ("(c) Dispersed", FIBRES, W_DISP),
           ("(d) Crossing", [component([1, 0, 0]), component([0, 1, 0])], [0.5, 0.5]),
           ("(e) Random", FIBRES, W_RAND),
           ("(f) Isotropic", [MD_STRONG * np.eye(3)], [1.0])]


def setup(ax):
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1), zoom=1.35); ax.set_axis_off(); ax.view_init(elev=22, azim=-55)


def draw_components(ax, comps, w):
    lmax = np.sqrt(STRONG[0])
    if len(comps) <= 2:
        shown = comps
    else:                                                        # 24 drawn, in proportion to the weights
        idx = np.random.default_rng(0).choice(len(comps), 24, replace=False, p=np.asarray(w))
        shown = [comps[i] for i in idx]
    # The restricted fibre: a solid intra-axonal stick drawn over a faint extra-axonal sheath, so that the
    # sheath does not tint it.
    sheathed = len(comps) == 2 and min(np.linalg.eigvalsh(T)[0] for T in comps) < 0.01 * E3
    ax.computed_zorder = not sheathed
    for j, T in enumerate(shown):
        lam, vec = np.linalg.eigh(T)
        E = GS @ (vec @ np.diag(np.sqrt(np.maximum(lam, 0.01 * E3)) / lmax) @ vec.T).T   # stick drawn thin
        col = COLORS[j % 2] if len(comps) == 2 else (COLORS[0] if fa_of(T) > 0 else "0.6")
        alpha = 0.8 if len(comps) == 1 else (0.5 if len(comps) == 2 else 0.18)
        stick = lam[0] < 0.01 * E3
        if sheathed:
            alpha = 1.0 if stick else 0.25
        ax.plot_surface(E[..., 0], E[..., 1], E[..., 2], color=col, alpha=alpha, linewidth=0, shade=True,
                        rasterized=True, zorder=2 if stick else 1)
    setup(ax)


NORM = plt.Normalize(0.2e-3, 1.8e-3)
_e, _a = np.radians(22), np.radians(-55)                         # the view of setup()
VIEW = np.array([np.cos(_e) * np.cos(_a), np.cos(_e) * np.sin(_a), np.sin(_e)])
ALPHA = 0.5                                                      # opacity of each layer of a surface
ALPHA_SEEN = 1 - (1 - ALPHA) ** 2                                # a closed surface puts two layers over the page
AXIS_LEN = 0.55                                                  # of the column's largest radius


def translucent(im, dpi):
    """Agg filter: make an opaquely rendered half-surface translucent as a whole, so that the overlapping edges of
    its faces cannot show as a lattice. A cap, not a product, because matplotlib applies the filter of a 3D
    collection twice."""
    im = im.copy()
    im[..., 3] = np.minimum(im[..., 3], ALPHA)
    return im, 0, 0


def draw_surface(ax, D, scale, label_axes=False):
    # No shading, so colour stays true to the colorbar. Depth comes from a small x, y, z triad at the origin, seen
    # through the translucent surface: the back half is drawn first, then the triad, then the front half.
    P = D[..., None] / scale * GS
    n = np.cross(np.gradient(P, axis=0), np.gradient(P, axis=1))
    flat = np.linalg.norm(n, axis=-1) < 1e-12                    # the poles of the grid: use the radial direction
    n[flat] = GS[flat]
    n *= np.sign((n * P).sum(-1, keepdims=True))                 # outward, since every profile is star-shaped
    front = n @ VIEW > 0
    ax.computed_zorder = False
    for half, z in ((~front, 1), (front, 3)):
        fc = cm.magma(NORM(D))
        fc[..., 3] = np.where(half, 1.0, 0.0)
        s = ax.plot_surface(*(P[..., i] for i in range(3)), facecolors=fc, rstride=1, cstride=1,
                            linewidth=0, antialiased=False, shade=False, rasterized=True, zorder=z)
        s.set_agg_filter(translucent)
    # Parallels and meridians every 45 degrees (every 10th grid line). The far parts lie behind the front half and
    # so show dimmed, which gives the curvature without touching the colour.
    grid = [P[i] for i in (10, 20, 30)] + [P[:, j] for j in range(0, 80, 10)]
    fmask = [front[i] for i in (10, 20, 30)] + [front[:, j] for j in range(0, 80, 10)]
    for line, f in zip(grid, fmask):
        for half, z in ((~f, 2), (f, 4)):
            keep = half | np.roll(half, 1) | np.roll(half, -1)   # overlap one point, so the halves meet
            seg = np.where(keep[:, None], line, np.nan)
            ax.plot(*seg.T, color=INK, lw=0.3, alpha=0.4, zorder=z)
    for k, name in enumerate("xyz"):
        tip = np.eye(3)[k] * AXIS_LEN
        ax.plot(*([0, t] for t in tip), color=INK, lw=0.6, zorder=2)
        if label_axes:
            ax.text(*(1.2 * tip), name, fontsize=6, color=INK, ha="center", va="center", zorder=4)
    setup(ax)


ROWS = ["FA", r"$J_{\ln}$", "fitted tensor", r"$J_{\ln}^{S}$", r"$\Delta_b J_{\ln}$"]
fig = plt.figure(figsize=(6.9, 5.5))
gs = fig.add_gridspec(5, len(CONFIGS), height_ratios=[1.0, 1.0, 1.0, 1.0, 0.55], hspace=0.0, wspace=0.0,
                      left=0.1, right=0.9, top=0.93, bottom=0.03)
row_axes = []
lines = ["config  FA_tensor  J_ln  J_ln_tensor  J_lnS  J_ln(3000)  delta_b_J_ln  delta_b_pct  ADC1(1500)  ADC1(3000)  "
         "J_lnS_one_component  lin_minus_iso  ratio  ratio_second_order   (b = 1500 unless stated)"]
for c, (title, comps, w) in enumerate(CONFIGS):
    S = signal(G, comps, w, B1)
    D = -np.log(S) / B1
    D2 = -np.log(signal(G, comps, w, B2)) / B2
    T = fit_tensor(D)
    fa, jln, jlt, jls, jl2 = fa_of(T), gap(D), gap(proj(G, T)), gap(S), gap(D2)
    db = jl2 - jln
    pct = 100 * db / jln if jln > 1e-9 else 0.0
    alike = all(np.allclose(np.linalg.eigvalsh(Tk), np.linalg.eigvalsh(comps[0])) for Tk in comps)
    jlc = gap(signal(G, comps[:1], [1.0], B1)) if alike else np.nan     # one component, when all are alike
    vc = v_lambda(comps[0]) if alike else 0.0
    mean_T = sum(wk * Tk for Tk, wk in zip(comps, w))
    ratio, ratio2 = (jls / jlc, v_lambda(mean_T) / vc) if vc > 0 else (np.nan, np.nan)
    s_iso = sum(wk * np.exp(-B1 * np.trace(Tk) / 3) for Tk, wk in zip(comps, w))   # isotropic encoding, same b
    mde = np.log(S.mean()) - np.log(s_iso)                                         # linear minus isotropic
    lines.append(f"{title:16s} {fa:.3f}  {jln:.4f}  {jlt:.4f}  {jls:.4f}  {jl2:.4f}  {db:+.4f}  {pct:+.1f}  "
                 f"{D.mean() / E3:.3f}  {D2.mean() / E3:.3f}  {jlc:.4f}  {mde:.4f}  {ratio:.3f}  {ratio2:.3f}")

    ax = fig.add_subplot(gs[0, c], projection="3d")
    draw_components(ax, comps, w)
    ax.set_title(title, pad=-4)
    Dp, Dp2 = (-np.log(signal(GS, comps, w, b)) / b for b in (B1, B2))
    scale = Dp.max()
    axes = [ax] + [fig.add_subplot(gs[r, c], projection="3d") for r in (1, 2, 3)]
    for r, (a3, Dsurf) in enumerate(zip(axes[1:], (proj(GS, T), Dp, Dp2))):
        draw_surface(a3, Dsurf, scale, label_axes=(c == 0 and r == 0))
    if c == 0:
        row_axes = axes

    ax = fig.add_subplot(gs[4, c]); ax.axis("off")
    vals = [f"{fa:.2f}", f"{jln:.3f}", f"{jlt:.3f}", f"{jls:.3f}",
            "0" if abs(db) < 1e-6 else f"{db:+.4f} ({pct:+.0f}%)".replace("-", "−")]
    for r, v in enumerate(vals):
        y = 0.95 - r * 0.2
        col = MUTED if r == 2 else INK
        ax.text(0.5, y, v, ha="center", va="top", fontsize=7, color=col)
        if c == 0:
            ax.text(-0.02, y, ROWS[r], ha="right", va="top", fontsize=7, color=col, transform=ax.transAxes)

for a3, label in zip(row_axes, ("Components", "Fitted tensor", "Directional ADC\n$b=1500$", "Directional ADC\n$b=3000$")):
    pos = a3.get_position()
    fig.text(pos.x0 - 0.005, (pos.y0 + pos.y1) / 2, label, rotation=90, ha="right", va="center", multialignment="center", fontsize=8)
pos = row_axes[2].get_position()
cax = fig.add_axes([0.915, pos.y0 + 0.02, 0.008, 0.2])
seen = cm.magma(np.linspace(0, 1, 256))                         # magma as the translucent surfaces show it on white
seen[:, :3] = ALPHA_SEEN * seen[:, :3] + (1 - ALPHA_SEEN)
cb = fig.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(0.2, 1.8),
                                    cmap=matplotlib.colors.ListedColormap(seen)), cax=cax)
cb.set_label("Directional diffusivity\n" r"($10^{-3}\,\mathrm{mm^2/s}$)", fontsize=6.5); cb.ax.tick_params(labelsize=6, length=2)

fig.savefig(FIGS / "fig_configurations.pdf", dpi=300, bbox_inches="tight")   # surfaces rasterized, text vector
fig.savefig(FIGS / "fig_configurations.png", dpi=200, bbox_inches="tight")
txt = "\n".join(lines)
(OUT / "fig_configurations.txt").write_text(txt + "\n", encoding="utf-8")
print(txt)
