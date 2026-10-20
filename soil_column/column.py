"""1D saturated soil column (2D plane strain, one element wide) in OpenSeesPy.

SSPquadUP u-p elements, water table at the surface (top drained), impermeable rigid base,
left/right nodes tied so the column deforms in simple shear. Reused by part 5 to build
the mini-surrogate dataset, so run_column() returns plain numpy arrays.
"""
import time

import numpy as np
import openseespy.opensees as ops

from .materials import G, P_ATM, RHO_W, g0_from_dr, pm4sand_args, rho_sat, void_ratio

K_FLUID = 2.2e6   # kPa, bulk modulus of water


def _nid(j, side):
    """Node id for row j (0 = base) and side 0/1 (left/right)."""
    return 2 * j + 1 + side


def build_column(H, dz, mat_args, e, perm, rho, gravity=True, alpha=None):
    """Mesh, boundary conditions, material and elements. mat_args = (type, *args).

    perm: hydraulic conductivity (m/s). OpenSees u-p elements want k / (rho_w g).
    """
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    nz = int(round(H / dz))
    for j in range(nz + 1):
        y = j * dz
        ops.node(_nid(j, 0), 0.0, y)
        ops.node(_nid(j, 1), dz, y)
    # Never put an sp on a dof that is also in an equalDOF (Transformation drops the MP).
    for s in (0, 1):
        ops.fix(_nid(0, s), 1, 1, 0)          # rigid base, impermeable
        ops.fix(_nid(nz, s), 0, 0, 1)         # drained surface, p = 0
    ops.equalDOF(_nid(0, 0), _nid(0, 1), 3)
    for j in range(1, nz):
        ops.equalDOF(_nid(j, 0), _nid(j, 1), 1, 2, 3)
    ops.equalDOF(_nid(nz, 0), _nid(nz, 1), 1, 2)

    ops.nDMaterial(mat_args[0], 1, *mat_args[1:])
    k = perm / (RHO_W * G)
    if alpha is None:          # SSPquadUP pressure stabilisation ~ h^2 / (4 (K + 4G/3))
        alpha = dz ** 2 / (4 * 2.0e5)
    b2 = -G if gravity else 0.0
    for j in range(nz):
        ops.element("SSPquadUP", j + 1, _nid(j, 0), _nid(j, 1), _nid(j + 1, 1), _nid(j + 1, 0),
                    1, 1.0, K_FLUID, RHO_W, k, k, e, alpha, 0.0, b2)
    return nz


def _transient(gamma=0.5, beta=0.25, tol=1e-5, algo="Newton"):
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormDispIncr", tol, 30)
    ops.algorithm(algo)
    ops.integrator("Newmark", gamma, beta)
    ops.analysis("Transient")


def pore_pressure(node):
    """SSPquadUP stores pore pressure in the velocity of dof 3."""
    return ops.nodeVel(node, 3)


def run_column(acc, dt, dr=0.45, H=10.0, dz=0.5, perm=1e-4, damping=0.01,
               t_post=10.0, dt_analysis=None, verbose=False):
    """Effective-stress site response of a uniform PM4Sand column on a rigid base.

    acc: base acceleration history (m/s^2) sampled at dt.
    Returns dict of arrays: t, depth (nodes), sigv0, ru [nt, nz_nodes], acc_surf (absolute),
    disp_surf (lateral, relative to base), settle (surface), gamma [nt, ne], tau [nt, ne].
    'ok' is False if the analysis failed to converge (caller should reject the run).
    """
    t0 = time.time()
    e, rho = void_ratio(dr), rho_sat(dr)
    nz = build_column(H, dz, ("PM4Sand", *pm4sand_args(dr)), e, perm, rho)
    elems = list(range(1, nz + 1))
    left = [_nid(j, 0) for j in range(nz + 1)]

    # --- gravity: elastic stage, then plastic stage (strongly dissipative Newmark) ---
    ops.updateMaterialStage("-material", 1, "-stage", 0)
    _transient(5 / 6, 4 / 9)
    if ops.analyze(20, 100.0) != 0:
        raise RuntimeError("gravity (elastic) failed")
    ops.updateMaterialStage("-material", 1, "-stage", 1)
    for el in elems:
        ops.setParameter("-val", 0, "-ele", el, "FirstCall", "1")
    if ops.analyze(20, 100.0) != 0:
        raise RuntimeError("gravity (plastic) failed")

    p_static = np.array([pore_pressure(n) for n in left])
    depth = H - np.array([ops.nodeCoord(n, 2) for n in left])
    sigv0 = (rho - RHO_W) * G * depth            # vertical effective stress at nodes
    uy0 = ops.nodeDisp(left[-1], 2)
    ops.setTime(0.0)
    ops.wipeAnalysis()

    # --- shaking ---
    dt_a = dt if dt_analysis is None else dt_analysis
    n_in = len(acc)
    t_end = n_in * dt + t_post
    ops.timeSeries("Path", 10, "-dt", dt, "-values", *acc, "-factor", 1.0)
    ops.pattern("UniformExcitation", 10, 1, "-accel", 10)
    # Rayleigh damping, `damping` ratio at 1 Hz and 10 Hz (current-tangent stiffness term)
    w1, w2 = 2 * np.pi * 1.0, 2 * np.pi * 10.0
    a0 = damping * 2 * w1 * w2 / (w1 + w2)
    a1 = damping * 2 / (w1 + w2)
    ops.rayleigh(a0, a1, 0.0, 0.0)
    _transient(0.6, 0.3025)

    nsteps = int(round(t_end / dt_a))
    t = np.zeros(nsteps + 1)
    ru = np.zeros((nsteps + 1, len(left)))
    acc_s = np.zeros(nsteps + 1)
    disp_s = np.zeros(nsteps + 1)
    settle = np.zeros(nsteps + 1)
    gam = np.zeros((nsteps + 1, nz))
    tau = np.zeros((nsteps + 1, nz))
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_sig = np.where(sigv0 > 0, 1.0 / sigv0, 0.0)
    ok = True
    for i in range(1, nsteps + 1):
        if ops.analyze(1, dt_a) != 0:
            # substep with a more robust algorithm before giving up
            _transient(0.6, 0.3025, tol=1e-4, algo="KrylovNewton")
            ok = all(ops.analyze(1, dt_a / 10) == 0 for _ in range(10))
            _transient(0.6, 0.3025)
            if not ok:
                if verbose:
                    print(f"  diverged at t = {ops.getTime():.2f} s")
                nsteps = i - 1
                break
        t[i] = ops.getTime()
        p = np.array([pore_pressure(n) for n in left])
        ru[i] = (p - p_static) * inv_sig
        ag = np.interp(t[i], np.arange(n_in) * dt, acc, right=0.0)
        acc_s[i] = ops.nodeAccel(left[-1], 1) + ag
        disp_s[i] = ops.nodeDisp(left[-1], 1)
        settle[i] = ops.nodeDisp(left[-1], 2) - uy0
        for k, el in enumerate(elems):
            s = ops.eleResponse(el, "stress")
            st = ops.eleResponse(el, "strain")
            tau[i, k] = s[2]
            gam[i, k] = st[2]
    sl = slice(0, nsteps + 1)
    out = dict(t=t[sl], depth=depth, sigv0=sigv0, ru=ru[sl], acc_surf=acc_s[sl], disp_surf=disp_s[sl],
               settle=settle[sl], gamma=gam[sl], tau=tau[sl],
               depth_el=H - (np.arange(nz) + 0.5) * dz, ok=ok, runtime=time.time() - t0,
               params=dict(dr=dr, H=H, dz=dz, perm=perm, damping=damping, dt=dt))
    return out


def tapered_sine(amp_g=0.15, freq=2.0, n_cycles=10, n_ramp=2, dt=0.005):
    """Uniform sine with cosine ramps (LEAP-style centrifuge input). Returns (acc m/s^2, dt)."""
    T = (n_cycles + 2 * n_ramp) / freq
    t = np.arange(0, T, dt)
    env = np.ones_like(t)
    tr = n_ramp / freq
    up, dn = t < tr, t > T - tr
    env[up] = 0.5 * (1 - np.cos(np.pi * t[up] / tr))
    env[dn] = 0.5 * (1 - np.cos(np.pi * (T - t[dn]) / tr))
    return amp_g * G * env * np.sin(2 * np.pi * freq * t), dt
