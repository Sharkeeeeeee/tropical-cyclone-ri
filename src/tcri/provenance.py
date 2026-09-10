"""Label provenance as a measurable quantity.

Outside the North Atlantic and East Pacific there is no routine aircraft
reconnaissance, so best-track intensity is estimated from infrared imagery by
the Dvorak technique -- the same imagery a vision model reads. Reconnaissance is
intermittent even where it exists, so this is a gradient, not a dichotomy.

Two properties make the gradient measurable: Dvorak assigns intensity through a
discrete CI table, and it holds the final T-number to about +1.0 per day (hard
caps 1.0/6 h, 1.5/12 h, 2.0/18 h, 2.5/24 h for T >= 4.0).

The CI-to-wind table below is the NHC convention. JTWC has used others, which
blurs the signal in the west Pacific rather than creating one.
"""
from __future__ import annotations

import numpy as np

# Dvorak CI number -> maximum sustained wind (kt), NHC convention.
CI_TO_KT = {
    1.0: 25, 1.5: 25, 2.0: 30, 2.5: 35, 3.0: 45, 3.5: 55, 4.0: 65,
    4.5: 77, 5.0: 90, 5.5: 102, 6.0: 115, 6.5: 127, 7.0: 140, 7.5: 155,
    8.0: 170,
}
# Best tracks are reported on a 5 kt grid, so the table values are snapped to it
# before asking whether an intensity "lands on" a Dvorak value.
DVORAK_KT = sorted({int(5 * round(v / 5)) for v in CI_TO_KT.values()})

TYPICAL_T_PER_DAY = 1.0
MAX_T_PER_24H = 2.5          # for T >= 4.0


def is_dvorak_value(v) -> np.ndarray:
    return np.isin(np.asarray(v), DVORAK_KT)


def ci_at(kt: float) -> float:
    """The CI number whose tabulated wind is closest to `kt`."""
    ks = np.array(list(CI_TO_KT.keys()))
    vs = np.array(list(CI_TO_KT.values()), dtype=float)
    return float(ks[np.argmin(np.abs(vs - kt))])


def wind_after_t_change(kt: float, dt: float) -> float:
    """Wind speed reached by changing the CI number by `dt` from `kt`."""
    ks = np.array(sorted(CI_TO_KT))
    target = ci_at(kt) + dt
    target = float(np.clip(target, ks.min(), ks.max()))
    # CI numbers are on a 0.5 grid; snap and read the table.
    return float(CI_TO_KT[min(CI_TO_KT, key=lambda k: abs(k - target))])


def dvorak_rate_ceiling(kt: float, dt: float = TYPICAL_T_PER_DAY) -> float:
    """Intensity change over 24 h implied by a CI change of `dt` from `kt`."""
    return wind_after_t_change(kt, dt) - float(kt)


def local_excess(values, grid) -> np.ndarray:
    """Each grid value's share divided by the mean share of its neighbours.

    Comparing raw frequencies between basins does not work: basins differ in how
    intense their storms get, and that difference swamps any quantization. This
    divides out the local level of the distribution, leaving only whether a
    value is over-represented relative to the ones on either side of it. A value
    of 1 means no spike.
    """
    v = np.asarray(values)
    cnt = np.array([(v == g).sum() for g in grid], dtype=float)
    if cnt.sum() == 0:
        return np.full(len(grid), np.nan)
    share = cnt / cnt.sum()
    out = np.full(len(grid), np.nan)
    for i in range(1, len(grid) - 1):
        nb = 0.5 * (share[i - 1] + share[i + 1])
        if nb > 0:
            out[i] = share[i] / nb
    return out


def quantization_contrast(values, grid, rng=None, n_boot: int = 0):
    """Mean local excess on Dvorak values minus mean excess off them.

    Positive means intensities cluster on the Dvorak table. Returns the point
    estimate and, if `n_boot` is given, a bootstrap 95% interval over cases.
    """
    v = np.asarray(values)
    grid = np.asarray(grid)
    on = np.isin(grid, DVORAK_KT)

    def contrast(sample):
        ex = local_excess(sample, grid)
        m = np.isfinite(ex)
        if (m & on).sum() < 3 or (m & ~on).sum() < 3:
            return np.nan
        return float(ex[m & on].mean() - ex[m & ~on].mean())

    point = contrast(v)
    if not n_boot:
        return point, (np.nan, np.nan)
    rng = rng or np.random.default_rng(0)
    boots = [contrast(rng.choice(v, size=len(v), replace=True))
             for _ in range(n_boot)]
    boots = np.array([b for b in boots if np.isfinite(b)])
    if boots.size < 10:
        return point, (np.nan, np.nan)
    return point, (float(np.percentile(boots, 2.5)),
                   float(np.percentile(boots, 97.5)))
