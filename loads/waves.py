"""Irregular wave loads on a monopile: JONSWAP sea state + linear wave theory + Morison.

Inertia uses the MacCamy-Fuchs solution per frequency component. It equals Morison inertia
with Cm = 2 for slender piles, and corrects the fact that Morison overpredicts inertia once the pile is a sizeable fraction of the wavelength, D/L > ~0.2).
Drag uses the Morison drag term on the total particle velocity. Kinematics are integrated
up to the mean water level (no free-surface stretching).

SI units inside (N, m, s); forces are returned in kN and kN*m to match OpenSees.
"""
import numpy as np
from scipy.special import jvp, yvp

G = 9.81
RHO_SEA = 1025.0     # kg/m^3


def jonswap(f, hs, tp, gamma=3.3):
    """One-sided JONSWAP variance density S(f) [m^2/Hz], scaled so 4 sqrt(m0) = hs exactly."""
    f = np.asarray(f, dtype=float)
    fp = 1.0 / tp
    sigma = np.where(f <= fp, 0.07, 0.09)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        pm = f ** -5 * np.exp(-1.25 * (fp / f) ** 4)
        peak = gamma ** np.exp(-((f - fp) ** 2) / (2 * sigma ** 2 * fp ** 2))
        s = np.where(f > 0, pm * peak, 0.0)
    m0 = np.trapezoid(s, f)
    return s * (hs ** 2 / 16.0) / m0


def wavenumber(omega, depth):
    """Solve the linear dispersion relation omega^2 = g k tanh(k d) (Newton, vectorised)."""
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    k = omega ** 2 / G                                   # deep-water start
    k = np.maximum(k, omega / np.sqrt(G * depth))        # shallow-water start is larger
    for _ in range(50):
        th = np.tanh(k * depth)
        f = G * k * th - omega ** 2
        df = G * th + G * k * depth * (1 - th ** 2)
        dk = f / df
        k -= dk
        if np.max(np.abs(dk / k)) < 1e-12:
            break
    return k


def maccamy_fuchs(ka):
    """Amplitude factor A and phase lag delta of the diffraction inertia force (radius a)."""
    j, y = jvp(1, ka), yvp(1, ka)
    return 1.0 / np.sqrt(j ** 2 + y ** 2), np.arctan2(j, y)


def sea_state_components(hs, tp, rng, gamma=3.3, f_max_factor=4.0, n_freq=256):
    """Random-phase discretisation of JONSWAP: frequencies, amplitudes, phases."""
    fp = 1.0 / tp
    df = (f_max_factor * fp - 0.5 * fp) / n_freq
    f = 0.5 * fp + (np.arange(n_freq) + rng.uniform(0, 1, n_freq)) * df   # jittered: no repeat period
    amp = np.sqrt(2 * jonswap(f, hs, tp, gamma) * df)
    phase = rng.uniform(0, 2 * np.pi, n_freq)
    return f, amp, phase


def wave_loads(t, hs, tp, rng, depth=30.0, diameter=8.0, cd=1.0, gamma=3.3,
               ramp=None, nz=40):
    """Irregular-wave load history on a vertical monopile.

    Returns dict: eta (m, at pile axis), force (kN, total horizontal at mudline),
    moment (kN*m, about mudline), force_inertia, force_drag (kN), kc (Keulegan-Carpenter no.).
    """
    t = np.asarray(t, dtype=float)
    f, amp, ph = sea_state_components(hs, tp, rng, gamma)
    w = 2 * np.pi * f
    k = wavenumber(w, depth)
    r = diameter / 2
    A, delta = maccamy_fuchs(k * r)

    env = np.ones_like(t)
    ramp = 2 * tp if ramp is None else ramp
    if ramp > 0:
        m = t < ramp
        env[m] = 0.5 * (1 - np.cos(np.pi * t[m] / ramp))

    arg = np.outer(t, w) + ph                     # [nt, nf]
    eta = env * (np.cos(arg) @ amp)

    # inertia (MacCamy-Fuchs), integrated analytically over depth:
    #   f(z) = (4 rho g a / k) A cosh(k(z+d))/cosh(kd) * (-sin(wt + ph - delta))
    #   int_{-d}^{0} cosh(k(z+d)) dz = sinh(kd)/k
    #   int_{-d}^{0} (z+d) cosh(k(z+d)) dz = (kd sinh(kd) - cosh(kd) + 1)/k^2
    kd = k * depth
    c = 4 * RHO_SEA * G * amp / k * A / np.cosh(kd)
    fi_amp = c * np.sinh(kd) / k
    mi_amp = c * (kd * np.sinh(kd) - np.cosh(kd) + 1) / k ** 2
    s = -np.sin(arg - delta)
    f_in = env * (s @ fi_amp)
    m_in = env * (s @ mi_amp)

    # drag: numerical depth integration of 0.5 rho Cd D u|u|
    z = np.linspace(-depth, 0.0, nz)              # z = 0 at mean water level
    lever = z + depth
    u_amp_base = amp * w / np.sinh(kd)            # u(z) = a w cosh(k(z+d))/sinh(kd) cos(...)
    cos_arg = np.cos(arg)
    f_dr = np.zeros_like(t)
    m_dr = np.zeros_like(t)
    u_max = 0.0
    for zi, li, dzi in zip(z, lever, _trap_weights(z)):
        u = env * (cos_arg @ (u_amp_base * np.cosh(k * (zi + depth))))
        fd = 0.5 * RHO_SEA * cd * diameter * u * np.abs(u)
        f_dr += fd * dzi
        m_dr += fd * li * dzi
        u_max = max(u_max, np.abs(u).max())

    kc = u_max * tp / diameter
    return dict(eta=eta, force=(f_in + f_dr) / 1e3, moment=(m_in + m_dr) / 1e3,
                force_inertia=f_in / 1e3, force_drag=f_dr / 1e3, kc=kc)


def _trap_weights(z):
    w = np.zeros_like(z)
    dz = np.diff(z)
    w[:-1] += dz / 2
    w[1:] += dz / 2
    return w
