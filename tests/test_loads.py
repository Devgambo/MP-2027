"""Checks for the part 4 load generator. Run:  .venv\\Scripts\\python.exe -m pytest tests -q"""
import numpy as np
import pytest
from scipy import signal

from loads import generate_loads, load_h5, save_h5
from loads.combined import MAX_STEEPNESS
from loads.earthquakes import (G, PEER_DIR, Record, process, read_at2, resample, response_spectrum,
                               scale_to_pga, significant_window, split_by_event, synthetic_library)
from loads.waves import RHO_SEA, jonswap, maccamy_fuchs, sea_state_components, wave_loads, wavenumber


# ------------------------------------------------------------------ waves
def test_jonswap_hs_and_peak():
    f = np.linspace(0.01, 1.0, 20000)
    s = jonswap(f, hs=5.0, tp=10.0)
    assert 4 * np.sqrt(np.trapezoid(s, f)) == pytest.approx(5.0, rel=1e-6)
    assert f[np.argmax(s)] == pytest.approx(0.1, abs=2e-3)


def test_dispersion_limits():
    d = 30.0
    w_deep = 2 * np.pi / 3.0            # short wave: deep water
    assert wavenumber(w_deep, d)[0] == pytest.approx(w_deep ** 2 / G, rel=1e-4)
    w_sh = 2 * np.pi / 200.0            # very long wave: shallow water
    assert wavenumber(w_sh, d)[0] == pytest.approx(w_sh / np.sqrt(G * d), rel=1e-3)
    w = np.linspace(0.2, 3, 50)
    k = wavenumber(w, d)
    assert np.allclose(G * k * np.tanh(k * d), w ** 2, rtol=1e-10)


def test_maccamy_fuchs_reduces_to_morison_cm2():
    ka = 1e-3
    A, delta = maccamy_fuchs(ka)
    assert A == pytest.approx(np.pi * ka ** 2 / 2, rel=1e-4)    # => Morison inertia with Cm = 2
    assert abs(delta) < 1e-5
    assert maccamy_fuchs(1.0)[0] < np.pi / 2                    # diffraction reduces inertia


def test_realised_hs_matches_target():
    t = np.arange(0, 3600, 0.25)
    w = wave_loads(t, hs=4.0, tp=9.0, rng=np.random.default_rng(1), ramp=0)
    assert 4 * w["eta"].std() == pytest.approx(4.0, rel=0.05)


def test_inertia_matches_morison_for_slender_pile():
    """Thin pile (ka << 1): MacCamy-Fuchs inertia force == Morison rho*Cm*A*du/dt, Cm = 2."""
    d, D, hs, tp = 30.0, 0.5, 3.0, 8.0
    t = np.arange(0, 120, 0.1)
    w = wave_loads(t, hs, tp, np.random.default_rng(7), depth=d, diameter=D, ramp=0)
    f, a, ph = sea_state_components(hs, tp, np.random.default_rng(7))
    om, k = 2 * np.pi * f, wavenumber(2 * np.pi * f, d)
    # depth-integrated du/dt: a w^2 sinh(kd)/(k sinh(kd)) * (-sin) = a w^2 / k * (-sin)
    dudt_int = -np.sin(np.outer(t, om) + ph) @ (a * om ** 2 / k)
    f_morison = RHO_SEA * 2.0 * np.pi * (D / 2) ** 2 * dudt_int / 1e3
    assert np.max(np.abs(w["force_inertia"] - f_morison)) < 0.01 * np.abs(f_morison).max()


# ------------------------------------------------------------------ earthquakes
def test_response_spectrum_against_lsim():
    rng = np.random.default_rng(0)
    rec = synthetic_library(1, seed=3)[0]
    acc, dt = rec.acc, rec.dt
    periods = np.array([0.1, 0.5, 1.0, 2.0])
    sa = response_spectrum(acc, dt, periods)
    t = np.arange(len(acc)) * dt
    for T, s in zip(periods, sa):
        w = 2 * np.pi / T
        sys = signal.lti([-1.0], [1, 2 * 0.05 * w, w ** 2])     # x'' + 2zw x' + w^2 x = -ag
        _, x, _ = signal.lsim(sys, acc, t)
        assert s == pytest.approx(w ** 2 * np.abs(x).max(), rel=0.01)
    assert sa[0] > 0 and rng is not None


def test_read_at2(tmp_path):
    vals = np.sin(np.linspace(0, 20, 1003)) * 0.2
    body = "\n".join("".join(f"{v:15.7E}" for v in vals[i:i + 5]) for i in range(0, len(vals), 5))
    p = tmp_path / "RSN123_TEST_H1.AT2"
    p.write_text("PEER NGA STRONG MOTION DATABASE RECORD\n"
                 "Test Quake, 1/1/2000, Some Station, H1\n"
                 "ACCELERATION TIME SERIES IN UNITS OF G\n"
                 f"NPTS=  {len(vals)}, DT=   .0050 SEC\n" + body + "\n")
    r = read_at2(p)
    assert r.dt == 0.005 and len(r.acc) == 1003
    assert np.allclose(r.acc, vals * G, atol=1e-6)
    assert r.event == "Test Quake" and r.meta["rsn"] == 123


def test_resample_keeps_signal():
    dt = 0.005
    t = np.arange(0, 10, dt)
    a = np.sin(2 * np.pi * 2.0 * t)
    b = resample(a, dt, 0.01)
    assert np.allclose(b[100:-100], np.sin(2 * np.pi * 2.0 * np.arange(len(b)) * 0.01)[100:-100], atol=1e-3)


def test_process_and_scale():
    rec = process(synthetic_library(1, seed=5)[0])
    s = scale_to_pga(rec, 0.25)
    assert s.pga == pytest.approx(0.25 * G, rel=1e-12)
    t5, t95 = significant_window(rec.acc, rec.dt)
    assert t5 < 3.0                                   # trimmed: strong part starts near the start
    disp = np.cumsum(np.cumsum(rec.acc) * rec.dt) * rec.dt
    assert abs(disp[-1]) < 0.02                       # baseline corrected: ends within 2 cm


def test_split_by_event_no_leakage():
    recs = [Record(acc=np.zeros(10), dt=0.01, name=f"r{i}", event=f"E{i % 7}") for i in range(30)]
    sp = split_by_event(recs)
    ev = {k: {r.event for r in v} for k, v in sp.items()}
    assert not (ev["train"] & ev["test"]) and not (ev["train"] & ev["val"]) and not (ev["val"] & ev["test"])
    assert sum(len(v) for v in sp.values()) == 30


def test_split_stratified_by_magnitude():
    """Every magnitude band reaches val and test; aftershock sequences stay together;
    record counts land near the 70/10/20 target even with very uneven events."""
    rng = np.random.default_rng(3)
    recs = []
    for k, mag in enumerate(np.r_[rng.uniform(5.5, 6.5, 15), rng.uniform(6.5, 7, 8), rng.uniform(7, 7.8, 8)]):
        for seq in range(2 if k < 3 else 1):                # 3 two-event sequences ("X-01", "X-02")
            for i in range(int(rng.integers(2, 30))):
                recs.append(Record(acc=np.zeros(10), dt=0.01, name=f"r{k}_{seq}_{i}",
                                   event=f"Quake {k}-0{seq + 1}", meta={"magnitude": mag}))
    sp = split_by_event(recs)
    band = lambda r: int(np.digitize(r.meta["magnitude"], (6.5, 7.0)))
    for v in sp.values():
        assert {band(r) for r in v} == {0, 1, 2}
    seqs = {k: {r.event.split("-")[0] for r in v} for k, v in sp.items()}
    assert not (seqs["train"] & seqs["test"]) and not (seqs["train"] & seqs["val"]) and not (seqs["val"] & seqs["test"])
    frac = np.array([len(sp[k]) for k in ("train", "val", "test")]) / len(recs)
    assert np.allclose(frac, [0.7, 0.1, 0.2], atol=0.05)


@pytest.mark.skipif(not list(PEER_DIR.glob("*.AT2")), reason="no PEER records downloaded")
def test_peer_library_split():
    from loads.earthquakes import load_peer
    recs = load_peer()
    assert all(np.isfinite(r.meta.get("magnitude", np.nan)) for r in recs), "record missing from CSVs"
    sp = split_by_event(recs)
    for k in ("val", "test"):
        assert max(r.meta["magnitude"] for r in sp[k]) >= 7.0, f"no M>=7 event in {k}"


# ------------------------------------------------------------------ combined
def test_generate_loads(tmp_path):
    cases = generate_loads(6, seed=11)
    again = generate_loads(6, seed=11)
    for c, d in zip(cases, again):
        assert np.array_equal(c.acc, d.acc) and np.array_equal(c.force, d.force)   # reproducible
        p = c.params
        n_eq = int(round(p["t_eq"] / p["dt"]))
        assert np.all(c.acc[:n_eq] == 0)                          # quake starts after the gap
        assert np.abs(c.acc).max() == pytest.approx(p["pga_g"] * G, rel=1e-9)
        assert p["hs"] / (G * p["tp"] ** 2 / (2 * np.pi)) <= MAX_STEEPNESS + 1e-9
        assert p["source"] != "peer" or 0.25 <= p["scale_factor"] <= 4.0
        assert len(c.t) == len(c.acc) == len(c.force) == len(c.moment)
    save_h5(cases, tmp_path / "loads.h5")
    back = load_h5(tmp_path / "loads.h5")
    assert np.allclose(back[2].force, cases[2].force, rtol=1e-5)
    assert back[2].params["record"] == cases[2].params["record"]


def test_split_cases_use_disjoint_events():
    tr = {c.params["event"] for c in generate_loads(20, seed=1, split="train")}
    te = {c.params["event"] for c in generate_loads(10, seed=2, split="test")}
    assert not tr & te


def test_peer_pipeline_end_to_end(tmp_path):
    """Fake .AT2 files -> load_peer -> process -> generate_loads; vertical files skipped."""
    from loads.earthquakes import load_peer
    rng = np.random.default_rng(0)
    for i, (ev, comp) in enumerate([("Quake A", "H1"), ("Quake A", "H2"), ("Quake B", "H1"),
                                    ("Quake B", "UP")]):
        vals = synthetic_library(1, seed=i)[0].acc / G * 0.5 + 1e-3 * rng.standard_normal(1)
        body = "\n".join("".join(f"{v:15.7E}" for v in vals[j:j + 5]) for j in range(0, len(vals), 5))
        (tmp_path / f"RSN{i + 1}_FAKE_{comp}.AT2").write_text(
            f"PEER NGA STRONG MOTION DATABASE RECORD\n{ev}, 1/1/2000, Station {i}, {comp}\n"
            f"ACCELERATION TIME SERIES IN UNITS OF G\nNPTS= {len(vals)}, DT= .0100 SEC\n{body}\n")
    hdr = ("Result ID, Spectral Ordinate, Record Sequence Number, Earthquake Name, Year, Magnitude, "
           "Mechanism, Rrup (km), Vs30 (m/sec), Lowest Useable Frequency (Hz)")
    (tmp_path / "_SearchResults.csv").write_text(
        f"PEER report\n\n{hdr}\n"
        + "".join(f'{i}, SRSS, {i}, "Quake", 2000, {6 + i / 10}, Reverse, 10, 500, 0.1\n' for i in range(1, 5))
        + "\n\n -- Scaled Spectra --\n")
    recs = load_peer(tmp_path)
    assert len(recs) == 3 and {r.event for r in recs} == {"Quake A", "Quake B"}
    assert [r.meta["magnitude"] for r in recs] == pytest.approx([6.1, 6.2, 6.3])
    cases = generate_loads(4, seed=0, records=[process(r) for r in recs])
    for c in cases:
        assert c.params["source"] == "peer"
        assert 0.25 <= c.params["scale_factor"] <= 4.0      # fake records ~0.15 g, targets 0.05-0.4 g
