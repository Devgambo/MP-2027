"""Checks for the part 6 pile model (static state and structure; the dynamic runs live in
pile_model/checks.py because they take minutes)."""
import numpy as np
import openseespy.opensees as ops
import pytest

from pile_model.model import E_STEEL, RHO_STEEL, Turbine, build, fixed_base_frequency, gravity


def test_fixed_base_frequency_matches_cantilever_formula():
    """Uniform cantilever with a tip mass: f = (1/2pi) sqrt(3EI / ((M + 0.24 m L) L^3))."""
    tb = Turbine(water_depth=0.0, platform=0.0, D=6.0, t_pile=0.03, D_tower=6.0, t_tower=0.03, hub=100.0)
    A, I = tb.tube(6.0, 0.03)
    L, m = 100.0, A * RHO_STEEL
    f_exact = np.sqrt(3 * E_STEEL * I / ((tb.m_rna + 0.24 * m * L) * L ** 3)) / (2 * np.pi)
    assert fixed_base_frequency(tb) == pytest.approx(f_exact, rel=0.03)


def test_default_turbine_frequency_in_10mw_range():
    assert 0.20 < fixed_base_frequency() < 0.30


@pytest.fixture(scope="module")
def gravity_state(tmp_path_factory):
    ops.logFile(str(tmp_path_factory.mktemp("ops") / "log.txt"), "-noEcho")
    pm = build(0.45, 1e-4)
    gravity(pm)
    return pm


def _stress(pm, i, j):
    ni = len(pm.x)
    s = ops.eleResponse(j * (ni - 1) + i + 1, "stress")
    depth = pm.mesh.H - 0.5 * (pm.y[j] + pm.y[j + 1])
    return -s[0], -s[1], s[2], depth


def test_free_field_gravity_stress(gravity_state):
    pm = gravity_state
    gam = pm.sigv0[0] / pm.mesh.H
    for j in (1, 10, 15):
        sh, sv, tau, d = _stress(pm, 0, j)                 # far from the pile
        assert sv == pytest.approx(gam * d, rel=0.03)
        assert 0.35 < sh / sv < 0.55                      # K0 of a normally consolidated sand
        assert abs(tau) < 0.05 * sv


def test_soil_does_not_hang_on_pile(gravity_state):
    """Next to the pile, vertical stress is not reduced (the soil does not arch onto the stiff
    pile during its own settlement). It is raised by skin friction carrying part of the turbine's
    weight: modestly at depth, more near the mudline (about 1.6x at 3.5 m)."""
    pm = gravity_state
    gam = pm.sigv0[0] / pm.mesh.H
    i = int(np.argmin(np.abs(pm.x + 5.5)))
    _, sv, _, d = _stress(pm, i, 10)                       # 8.5 m depth
    assert 0.95 < sv / (gam * d) < 1.25
    _, sv, _, d = _stress(pm, i, 15)                       # 3.5 m depth
    assert 0.95 < sv / (gam * d) < 2.0


def test_structure_settles_and_stays_upright(gravity_state):
    pm = gravity_state
    assert 18e3 < pm.structure_weight < 25e3              # kN, 10 MW class
    v, th = ops.nodeDisp(pm.cap, 2), ops.nodeDisp(pm.cap, 3)
    assert -0.05 < v < 0                                   # settles, by centimetres at most
    assert abs(th) < 1e-8                                  # symmetric: no tilt from gravity


def test_rest_stays_at_rest(gravity_state):
    """With no waves and no shaking, a dynamic run must not move: catches gravity loads that
    switch off when the clock restarts at 0 (a real bug found in part 6)."""
    from pile_model.model import run
    r = run(np.zeros(300), 0.01, pm=gravity_state)
    ru = np.where(r["mask"], r["ru"], np.nan)
    assert np.nanmax(np.abs(ru)) < 1e-6
    assert np.nanmax(np.abs(r["tilt"])) < 1e-8 and np.nanmax(np.abs(r["cap_v"])) < 1e-8
