"""2D plane-strain monopile + tower in saturated sand under waves and an earthquake (part 6).

Units: kN, m, s, kPa, t (Mg). x = horizontal (0 at the pile axis), y = up (0 at the model base).

Modelling choices (see docs/part6_pile_model.md for the reasoning):
- Plane strain with an out-of-plane thickness equal to the pile diameter D (an 8 m slice).
  The pile gets its full bending stiffness and the full wave force; soil resistance outside the
  slice (3D flow around the pile) is missing, so tilt is over-predicted (conservative).
- Soil: SSPquadUP u-p elements with PM4Sand (calibrated in part 3), same as the soil column.
- Pile below the mudline: the steel tube plus its soil plug as a stiff elastic block, 8 m wide,
  whose bending stiffness equals the steel pile's (E_eq = EI_pile / (t w^3 / 12)). Bonded to the
  soil horizontally; vertically through friction-limited springs (no gapping).
- Above the mudline: elastic beam-columns (P-Delta) for the pile up to the platform and the
  tower, with the rotor-nacelle assembly (RNA) as a lumped mass on top. Submerged length carries
  internal water and hydrodynamic added mass (Ca = 1). Generic 10 MW-class turbine (DTU 10 MW).
- The beam meets the block through a rigid cap (stiff arms to every block top node), so the
  block's rotation is the pile tilt.
- Wave force and mudline moment from part 4 act at the cap; the earthquake shakes the rigid base.
- Lateral boundaries tied (periodic) so far-field soil moves like a free-field column; base rigid
  and impermeable; mudline drained.
"""
import time
from dataclasses import dataclass, field

import numpy as np
import openseespy.opensees as ops

from soil_column.materials import G, RHO_W, pm4sand_args, rho_sat, void_ratio

K_FLUID = 2.2e6            # kPa
E_STEEL = 2.1e8            # kPa
RHO_STEEL = 8.5            # t/m^3, includes flanges, paint, secondary steel
RHO_SEA = 1.025
K_IFACE = 1.0e6            # kN/m per node pair, vertical pile-soil interface spring stiffness
SKIN_K, SKIN_TAN_DELTA = 0.8, 0.40   # skin friction tau_max = K sigma'v tan(delta), delta = 22 deg (API-style)


@dataclass
class Turbine:
    """Generic 10 MW-class monopile turbine (DTU 10 MW reference values, rounded)."""
    D: float = 8.0               # pile diameter (m)
    t_pile: float = 0.08         # pile wall (m)
    L_embed: float = 24.0        # embedded length (m), L/D = 3
    water_depth: float = 30.0
    platform: float = 10.0       # pile-tower interface above sea level (m)
    hub: float = 119.0           # hub height above sea level (m)
    D_tower: float = 7.5         # average tower diameter (m)
    t_tower: float = 0.035       # average tower wall (m), fixed-base f1 = 0.226 Hz
    m_rna: float = 674.0         # rotor + nacelle (t)

    def tube(self, D, t):
        A = np.pi * D * t
        I = np.pi / 8 * D ** 3 * t
        return A, I


@dataclass
class Mesh:
    H: float = 30.0                                      # soil depth (m)
    x: tuple = (-50, -38, -28, -20, -14, -10, -7, -5.5, -4, -2, 0, 2, 4, 5.5, 7, 10, 14, 20, 28, 38, 50)
    y: tuple = (0, 3, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)
    stab_G: float = 2.0e5                                # shear modulus in the u-p stabilisation alpha = h^2/(4G)


@dataclass
class PileModel:
    """Node/element bookkeeping filled by build()."""
    soil_nodes: dict = field(default_factory=dict)       # (i, j) -> tag
    soil_eles: list = field(default_factory=list)
    block_eles: list = field(default_factory=list)
    beam_nodes: list = field(default_factory=list)       # cap ... hub (bottom to top)
    beam_eles: list = field(default_factory=list)
    cap: int = 0
    hub: int = 0
    sigv0: np.ndarray = None                             # free-field sigma'v at each node row
    mask: np.ndarray = None                              # [nj, ni] True where ru is meaningful (soil)
    thickness: float = 8.0


def build(dr=0.45, perm=1e-4, turbine=None, mesh=None):
    tb, me = turbine or Turbine(), mesh or Mesh()
    x, y = np.array(me.x, float), np.array(me.y, float)
    H, w2, t_out = me.H, tb.D / 2, tb.D
    y_tip = H - tb.L_embed
    assert np.isclose(x, w2).any() and np.isclose(y, y_tip).any(), "mesh must have lines at the pile edge and tip"
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    pm = PileModel(thickness=t_out)
    ni, nj = len(x), len(y)
    tag = 0
    for j in range(nj):
        for i in range(ni):
            tag += 1
            ops.node(tag, x[i], y[j])
            pm.soil_nodes[(i, j)] = tag
    N = pm.soil_nodes                                     # grid nodes; inside the footprint they belong to the block
    j_tip = int(np.argmin(np.abs(y - y_tip)))
    # The pile wall gets its own soil nodes: soil and pile share horizontal motion (soil pushes on
    # the pile) but may slide vertically during the soil's gravity stage; vertical springs are added
    # afterwards (gravity()), so the soil does not hang on the pile. The wall is impermeable.
    side = {}
    for j in range(j_tip + 1, nj):
        for i in range(ni):
            if np.isclose(abs(x[i]), w2):
                tag += 1
                ops.node(tag, x[i], y[j])
                side[(i, j)] = tag
                ops.equalDOF(N[(i, j)], tag, 1)
    pm.side_pairs = [(N[k], v) for k, v in side.items()]

    def soil_node(i, j):
        return side.get((i, j), N[(i, j)])

    def in_block(i, j):            # element (i, j) spans x[i]..x[i+1], y[j]..y[j+1]
        return x[i] >= -w2 - 1e-9 and x[i + 1] <= w2 + 1e-9 and y[j] >= y_tip - 1e-9

    # boundary conditions
    for i in range(ni):
        ops.fix(N[(i, 0)], 1, 1, 0)                         # rigid impermeable base
        ops.fix(N[(i, nj - 1)], 0, 0, 1)                    # drained mudline (soil and the pile's soil plug)
    for n in (side[k] for k in side if k[1] == nj - 1):
        ops.fix(n, 0, 0, 1)
    for j in range(1, nj):                                  # periodic lateral boundaries
        ops.equalDOF(N[(0, j)], N[(ni - 1, j)], 1, 2)

    # materials: 1 = PM4Sand, 2 = elastic pile+plug block
    e, rho = void_ratio(dr), rho_sat(dr)
    ops.nDMaterial("PM4Sand", 1, *pm4sand_args(dr))
    A_p, I_p = tb.tube(tb.D, tb.t_pile)
    E_eq = E_STEEL * I_p / (t_out * tb.D ** 3 / 12)
    ops.nDMaterial("ElasticIsotropic", 2, E_eq, 0.25, rho)
    k_soil = perm / (RHO_W * G)                             # the plug is soil, so same permeability
    etag = 0
    for j in range(nj - 1):
        for i in range(ni - 1):
            etag += 1
            blk = in_block(i, j)
            corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            nodes = [N[c] for c in corners] if blk else [soil_node(*c) for c in corners]
            h = max(x[i + 1] - x[i], y[j + 1] - y[j])      # largest side: thin elements still checkerboard
            ops.element("SSPquadUP", etag, *nodes, 2 if blk else 1, t_out, K_FLUID, RHO_W, k_soil, k_soil,
                        e, h ** 2 / (4 * me.stab_G), 0.0, -G)
            (pm.block_eles if blk else pm.soil_eles).append(etag)
    pm.ru_nodes = [[int(soil_node(i, j)) for i in range(ni)] for j in range(nj)]

    # structure above the mudline: cap node, beam up to the hub
    ops.geomTransf("PDelta", 1)
    ops.geomTransf("Linear", 2)
    tag = 100000
    cap = tag
    ops.node(cap, 0.0, H)
    # rigid arms from the cap to every block top node
    big_A, big_I = 100 * A_p, 100 * I_p
    arm_eles = []
    for i in range(ni):
        if abs(x[i]) <= w2 + 1e-9:
            if np.isclose(x[i], 0.0):
                ops.equalDOF(N[(i, nj - 1)], cap, 1, 2)
                continue
            tag += 1
            ops.node(tag, x[i], H)
            ops.equalDOF(N[(i, nj - 1)], tag, 1, 2)
            etag += 1
            ops.element("elasticBeamColumn", etag, cap, tag, big_A, E_STEEL, big_I, 2)
            arm_eles.append(etag)
    # beam nodes: mudline -> sea level -> platform -> hub
    z_mud = -tb.water_depth
    levels = np.unique(np.r_[np.linspace(z_mud, 0, 7), np.linspace(0, tb.platform, 3),
                             np.linspace(tb.platform, tb.hub, 12)])
    A_t, I_t = tb.tube(tb.D_tower, tb.t_tower)
    m_water = RHO_SEA * np.pi * (tb.D - 2 * tb.t_pile) ** 2 / 4 + RHO_SEA * np.pi * tb.D ** 2 / 4   # inside + added
    prev = cap
    pm.beam_nodes = [cap]
    for k, z in enumerate(levels[1:], 1):
        tag += 1
        ops.node(tag, 0.0, H + (z - z_mud))
        zm = 0.5 * (z + levels[k - 1])
        if zm < tb.platform:
            A, I, m = A_p, I_p, A_p * RHO_STEEL + (m_water if zm < 0 else 0.0)
        else:
            A, I, m = A_t, I_t, A_t * RHO_STEEL
        etag += 1
        ops.element("elasticBeamColumn", etag, prev, tag, A, E_STEEL, I, 1, "-mass", m)
        pm.beam_eles.append(etag)
        pm.beam_nodes.append(tag)
        prev = tag
    pm.cap, pm.hub = cap, prev
    ops.mass(pm.hub, tb.m_rna, tb.m_rna, 0.0)
    pm.arm_eles = arm_eles
    pm.next_ele = etag + 1

    pm.z_mud = z_mud
    depth = H - y
    pm.sigv0 = (rho - RHO_W) * G * depth                 # free-field effective vertical stress per row
    pm.mask = np.ones((nj, ni), bool)
    for j in range(nj):
        for i in range(ni):
            if abs(x[i]) < w2 - 1e-9 and y[j] > y_tip + 1e-9:
                pm.mask[j, i] = False                     # inside the block: not soil
    pm.x, pm.y, pm.turbine, pm.mesh = x, y, tb, me
    pm.params = dict(dr=dr, perm=perm, E_eq=E_eq, n_soil=len(pm.soil_eles), n_block=len(pm.block_eles))
    return pm


def _structure_weight(pm):
    """Beam self-weight + RNA as nodal loads (soil weight is an element body force)."""
    tb, H, z_mud = pm.turbine, pm.mesh.H, pm.z_mud
    A_p, _ = tb.tube(tb.D, tb.t_pile)
    A_t, _ = tb.tube(tb.D_tower, tb.t_tower)
    t0 = ops.getTime()                                    # ramp the weight on over 2000 s, then hold
    ops.timeSeries("Path", 1, "-time", t0, t0 + 2000.0, t0 + 1e9, "-values", 0.0, 1.0, 1.0)
    ops.pattern("Plain", 1, 1)
    w_nodes = np.zeros(len(pm.beam_nodes))
    for k in range(len(pm.beam_eles)):
        z0 = ops.nodeCoord(pm.beam_nodes[k], 2)
        z1 = ops.nodeCoord(pm.beam_nodes[k + 1], 2)
        zm = 0.5 * (z0 + z1) - H + z_mud
        m = A_p * RHO_STEEL if zm < tb.platform else A_t * RHO_STEEL      # added water mass has no weight
        w_nodes[k] += 0.5 * m * (z1 - z0) * G
        w_nodes[k + 1] += 0.5 * m * (z1 - z0) * G
    w_nodes[-1] += tb.m_rna * G
    for n, wn in zip(pm.beam_nodes, w_nodes):
        ops.load(n, 0.0, -wn, 0.0)
    pm.weight_loads = list(zip(pm.beam_nodes, w_nodes))
    return float(w_nodes.sum())


def _transient(gamma=0.5, beta=0.25, tol=1e-4, algo="Newton"):
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormDispIncr", tol, 30)
    ops.algorithm(algo)
    ops.integrator("Newmark", gamma, beta)
    ops.analysis("Transient")


def gravity(pm, slip=True):
    """Static stages, all with the soil elastic (applying 19 MN to plastic PM4Sand in one go does
    not converge), then the switch to plastic PM4Sand, as in the soil column:
    1. soil self-weight; the pile wall may slide vertically, so the soil does not hang on the pile
    2. bond the wall vertically (springs that start at zero force), add the turbine's weight
    3. PM4Sand plastic stage."""
    ops.updateMaterialStage("-material", 1, "-stage", 0)
    _transient(5 / 6, 4 / 9)
    if ops.analyze(20, 100.0) != 0:
        raise RuntimeError("gravity (soil) failed")
    # friction-limited: each spring slips at tau_max x (tributary wall height) x slice thickness
    pm.iface_eles = []
    ys = np.array(sorted({ops.nodeCoord(b, 2) for b, _ in pm.side_pairs} | {pm.mesh.H - pm.turbine.L_embed}))
    gam_eff = (rho_sat(pm.params["dr"]) - RHO_W) * G
    for k, (blk, dup) in enumerate(pm.side_pairs):
        yk = ops.nodeCoord(blk, 2)
        i = int(np.argmin(np.abs(ys - yk)))
        h = 0.5 * ((ys[min(i + 1, len(ys) - 1)] - ys[i]) + (ys[i] - ys[max(i - 1, 0)]))
        sig_v = gam_eff * max(pm.mesh.H - yk, 0.25)          # avoid zero capacity at the mudline node
        f_max = SKIN_K * SKIN_TAN_DELTA * sig_v * h * pm.thickness
        d = ops.nodeDisp(dup, 2) - ops.nodeDisp(blk, 2)
        if slip:
            ops.uniaxialMaterial("Steel01", 1000 + k, f_max, K_IFACE, 0.01)     # 1% stiffness after slip
        else:
            ops.uniaxialMaterial("Elastic", 1000 + k, K_IFACE)
        ops.uniaxialMaterial("InitStrainMaterial", 2000 + k, 1000 + k, -d)
        ops.element("zeroLength", pm.next_ele, blk, dup, "-mat", 2000 + k, "-dir", 2)
        pm.iface_eles.append(pm.next_ele)
        pm.next_ele += 1
    pm.structure_weight = _structure_weight(pm)
    _transient(5 / 6, 4 / 9, algo="KrylovNewton")       # plain Newton stalls at the springs' slip corner
    if ops.analyze(40, 100.0) != 0:
        raise RuntimeError("gravity (structure) failed")
    ops.updateMaterialStage("-material", 1, "-stage", 1)
    for el in pm.soil_eles:
        ops.setParameter("-val", 0, "-ele", el, "FirstCall", "1")
    _transient(5 / 6, 4 / 9)
    if ops.analyze(40, 100.0) != 0:
        raise RuntimeError("gravity (plastic) failed")
    # The weight was ramped on by a time-based curve; runs restart the clock at 0, where that curve
    # is zero. Swap it for a constant pattern with the same loads, so the weight stays on.
    ops.remove("loadPattern", 1)
    ops.timeSeries("Constant", 5)
    ops.pattern("Plain", 5, 5)
    for n, wn in pm.weight_loads:
        ops.load(n, 0.0, -wn, 0.0)
    pm.p_static = np.array([[ops.nodeVel(n, 3) for n in row] for row in pm.ru_nodes])
    pm.disp0 = {n: (ops.nodeDisp(n, 1), ops.nodeDisp(n, 2), ops.nodeDisp(n, 3)) for n in pm.beam_nodes}
    ops.setTime(0.0)
    ops.wipeAnalysis()


def fixed_base_frequency(turbine=None):
    """First natural frequency (Hz) of the pile + tower + RNA clamped at the mudline (no soil).
    Built as a separate beam-only model; the soil-structure system must come out lower."""
    tb = turbine or Turbine()
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.geomTransf("Linear", 1)
    A_p, I_p = tb.tube(tb.D, tb.t_pile)
    A_t, I_t = tb.tube(tb.D_tower, tb.t_tower)
    m_water = RHO_SEA * np.pi * (tb.D - 2 * tb.t_pile) ** 2 / 4 + RHO_SEA * np.pi * tb.D ** 2 / 4
    z = np.unique(np.r_[np.linspace(-tb.water_depth, 0, 7), np.linspace(0, tb.platform, 3),
                        np.linspace(tb.platform, tb.hub, 12)])
    for k, zk in enumerate(z):
        ops.node(k + 1, 0.0, zk)
    ops.fix(1, 1, 1, 1)
    for k in range(1, len(z)):
        zm = 0.5 * (z[k] + z[k - 1])
        A, I, m = (A_p, I_p, A_p * RHO_STEEL + (m_water if zm < 0 else 0)) if zm < tb.platform else (A_t, I_t, A_t * RHO_STEEL)
        ops.element("elasticBeamColumn", k, k, k + 1, A, E_STEEL, I, 1, "-mass", m)
    ops.mass(len(z), tb.m_rna, tb.m_rna, 1e-6)
    lam = ops.eigen("-fullGenLapack", 1)[0]
    ops.wipe()
    return float(np.sqrt(lam) / (2 * np.pi))


def _rayleigh(f1, f2, zeta):
    w1, w2 = 2 * np.pi * f1, 2 * np.pi * f2
    return zeta * 2 * w1 * w2 / (w1 + w2), zeta * 2 / (w1 + w2)


def run(acc, dt, force=None, moment=None, dr=0.45, perm=1e-4, t_post=0.0, dt_waves=0.05,
        dt_quake=0.01, out_dt=0.05, turbine=None, mesh=None, verbose=False, pm=None,
        algo="Newton", tol=1e-4, slip=True):
    """One combined run. acc, force (kN), moment (kN m) share the time step dt (part 4 LoadCase).

    Time stepping: dt_waves while there is no shaking (waves only), dt_quake from 1 s before the
    shaking starts until 2 s after it ends, then dt_waves again. Returns arrays on an out_dt grid.
    """
    t0 = time.time()
    if pm is None:
        pm = build(dr, perm, turbine, mesh)
        gravity(pm, slip=slip)
    n_in = len(acc)
    force = np.zeros(n_in) if force is None else np.asarray(force)
    moment = np.zeros(n_in) if moment is None else np.asarray(moment)
    t_in = np.arange(n_in) * dt
    shaking = np.abs(acc) > 1e-6
    t_on = t_in[shaking][0] - 1.0 if shaking.any() else np.inf
    t_off = t_in[shaking][-1] + 2.0 if shaking.any() else -np.inf
    t_end = t_in[-1] + t_post

    ops.timeSeries("Path", 10, "-dt", dt, "-values", *acc)
    ops.pattern("UniformExcitation", 10, 1, "-accel", 10)
    ops.timeSeries("Path", 11, "-dt", dt, "-values", *force)
    ops.timeSeries("Path", 12, "-dt", dt, "-values", *moment)
    ops.pattern("Plain", 11, 11)
    ops.load(pm.cap, 1.0, 0.0, 0.0)
    ops.pattern("Plain", 12, 12)
    ops.load(pm.cap, 0.0, 0.0, 1.0)
    # damping: soil 1% at 1 and 10 Hz (as the soil column); structure 1% at 0.25 and 2 Hz
    a0, a1 = _rayleigh(1.0, 10.0, 0.01)
    ops.region(1, "-ele", *pm.soil_eles, *pm.block_eles, "-rayleigh", a0, a1, 0.0, 0.0)
    b0, b1 = _rayleigh(0.25, 2.0, 0.01)
    ops.region(2, "-ele", *pm.beam_eles, "-rayleigh", b0, b1, 0.0, 0.0)
    ops.region(3, "-node", pm.hub, "-rayleigh", b0, 0.0, 0.0, 0.0)
    _transient(0.6, 0.3025, tol=tol, algo=algo)

    nj, ni = pm.mask.shape
    nodes = pm.ru_nodes
    with np.errstate(divide="ignore"):
        inv_sig = np.where(pm.sigv0 > 0, 1.0 / pm.sigv0, 0.0)[:, None]
    n_out = int(np.floor(t_end / out_dt + 1e-9)) + 1
    out = {k: np.full(n_out, np.nan) for k in ("tilt", "cap_u", "cap_v", "hub_u", "acc_in", "force", "moment")}
    ru = np.full((n_out, nj, ni), np.nan, np.float32)
    k_out, ok, steps, n_sub = 0, True, 0, 0

    def record(k):
        th = ops.nodeDisp(pm.cap, 3) - pm.disp0[pm.cap][2]
        out["tilt"][k] = np.degrees(th)
        out["cap_u"][k] = ops.nodeDisp(pm.cap, 1)
        out["cap_v"][k] = ops.nodeDisp(pm.cap, 2) - pm.disp0[pm.cap][1]
        out["hub_u"][k] = ops.nodeDisp(pm.hub, 1)
        tt = ops.getTime()
        out["acc_in"][k] = np.interp(tt, t_in, acc, right=0.0)
        out["force"][k] = np.interp(tt, t_in, force, right=0.0)
        out["moment"][k] = np.interp(tt, t_in, moment, right=0.0)
        p = np.array([[ops.nodeVel(n, 3) for n in row] for row in nodes])
        ru[k] = (p - pm.p_static) * inv_sig

    record(0)
    k_out = 1
    while k_out < n_out:
        t_now = ops.getTime()
        h = dt_quake if t_on - 1e-9 <= t_now < t_off else dt_waves
        t_next = k_out * out_dt
        h = min(h, t_next - t_now) if t_next - t_now > 1e-9 else h
        if ops.analyze(1, h) != 0:
            _transient(0.6, 0.3025, tol=max(tol, 1e-3), algo="KrylovNewton")
            sub = 10
            ok = all(ops.analyze(1, h / sub) == 0 for _ in range(sub))
            n_sub += 1
            _transient(0.6, 0.3025, tol=tol, algo=algo)
            if not ok:
                if verbose:
                    print(f"  diverged at t = {ops.getTime():.2f} s")
                break
        steps += 1
        if ops.getTime() >= t_next - 1e-6:
            record(k_out)
            k_out += 1
    res = dict(t=np.arange(n_out) * out_dt, ru=ru, ok=ok, steps=steps, n_substepped=n_sub, runtime=time.time() - t0,
               x=pm.x, y=pm.y, depth=pm.mesh.H - pm.y, mask=pm.mask, params=dict(pm.params, dt=dt), **out)
    return res
