"""Single-element undrained (constant-volume) cyclic direct simple shear test on PM4Sand.

Constant-volume DSS is the standard way to calibrate PM4Sand: the drop in vertical
effective stress equals the excess pore pressure an undrained test would build up.
Liquefaction is taken as 3% single-amplitude shear strain (Idriss & Boulanger 2008).
"""
import numpy as np
import openseespy.opensees as ops

from .materials import P_ATM, pm4sand_args


def _setup_static(ctrl_node=None, ctrl_dof=None, du=None):
    ops.wipeAnalysis()
    # Lagrange: Transformation silently drops the equalDOF once an sp hits the retained node
    ops.constraints("Lagrange")
    ops.numberer("RCM")
    ops.system("FullGeneral")
    ops.test("NormDispIncr", 1e-9, 100)
    ops.algorithm("Newton")
    if ctrl_node is None:
        ops.integrator("LoadControl", 0.1)
    else:
        ops.integrator("DisplacementControl", ctrl_node, ctrl_dof, du)
    ops.analysis("Static")


def _substep(du, n=10):
    """Retry a failed increment as n smaller ones with a more forgiving algorithm."""
    _setup_static(3, 1, du / n)
    ops.algorithm("KrylovNewton")
    ok = all(ops.analyze(1) == 0 for _ in range(n))
    _setup_static(3, 1, du)
    return ok


def cyclic_dss(dr, csr, hpo=None, sigv=P_ATM, gamma_liq=0.03, max_cycles=100, record=False):
    """Run a stress-controlled cyclic DSS. Returns dict with cycles to liquefaction (inf if none)."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    for i, (x, y) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)], 1):
        ops.node(i, x, y)
    ops.fix(1, 1, 1)
    ops.fix(2, 1, 1)
    ops.equalDOF(3, 4, 1, 2)          # top moves as a rigid cap -> simple shear kinematics
    ops.nDMaterial("PM4Sand", 1, *pm4sand_args(dr, hpo))
    ops.element("SSPquad", 1, 1, 2, 3, 4, 1, "PlaneStrain", 1.0)

    # 1) K0 consolidation to sigv, elastic stage
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(3, 0.0, -sigv / 2)
    ops.load(4, 0.0, -sigv / 2)
    ops.updateMaterialStage("-material", 1, "-stage", 0)
    _setup_static()
    if ops.analyze(10) != 0:
        raise RuntimeError("consolidation failed")
    ops.loadConst("-time", 0.0)

    # 2) switch to plastic, lock height (constant volume), shear the cap
    ops.updateMaterialStage("-material", 1, "-stage", 1)
    ops.setParameter("-val", 0, "-ele", 1, "FirstCall", "1")
    # sp at the consolidated height (ops.fix would impose zero and unload the element)
    ops.timeSeries("Constant", 3)
    ops.pattern("Plain", 3, 3)
    ops.sp(3, 2, ops.nodeDisp(3, 2))
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    ops.load(3, 1.0, 0.0)

    tau_target = csr * sigv
    sig0 = -ops.eleResponse(1, "stress")[1]
    direction, reversals, gamma = 1, 0, 0.0
    hist = []
    _setup_static(3, 1, 2e-6)
    small = True
    while reversals < 2 * max_cycles:
        if small and abs(gamma) > 0.003:
            _setup_static(3, 1, 1e-5 * direction)
            small = False
        if ops.analyze(1) != 0 and not _substep(direction * (2e-6 if small else 1e-5)):
            return {"N": np.nan, "failed": True, "hist": np.array(hist) if record else None}
        gamma = ops.nodeDisp(3, 1)
        s = ops.eleResponse(1, "stress")
        tau, sv = s[2], -s[1]
        if record:
            hist.append((gamma, tau, sv))
        if abs(gamma) >= gamma_liq:
            n = (reversals + 1) / 2.0    # half-cycles completed, counting the current one
            return {"N": n, "failed": False, "ru": 1 - sv / sig0,
                    "hist": np.array(hist) if record else None}
        if tau * direction >= tau_target:
            direction *= -1
            reversals += 1
            _setup_static(3, 1, (2e-6 if small else 1e-5) * direction)
    return {"N": np.inf, "failed": False, "ru": 1 - sv / sig0,
            "hist": np.array(hist) if record else None}
