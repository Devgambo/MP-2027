"""PM4Sand parameters and the Idriss & Boulanger (2008) liquefaction-resistance target.

Units throughout the soil_column package: kN, m, s, kPa, Mg/m^3 (t/m^3).
"""
import json
import math
from pathlib import Path

P_ATM = 101.3          # kPa
G = 9.81               # m/s^2
RHO_W = 1.0            # Mg/m^3
GS = 2.67              # specific gravity of sand grains
E_MAX, E_MIN = 0.80, 0.50   # PM4Sand defaults

CALIB_FILE = Path(__file__).with_name("pm4sand_calibration.json")


def n160_from_dr(dr):
    """Dr = sqrt((N1)60 / 46)  (Idriss & Boulanger 2008)."""
    return 46.0 * dr ** 2


def g0_from_dr(dr):
    """PM4Sand shear-modulus coefficient, G0 = 167 sqrt((N1)60 + 2.5) (Boulanger & Ziotopoulou 2017)."""
    return 167.0 * math.sqrt(n160_from_dr(dr) + 2.5)


def crr_m75(dr):
    """Clean-sand CRR for M7.5 and sigma'v = 1 atm (Idriss & Boulanger 2008, SPT form)."""
    n = n160_from_dr(dr)
    return math.exp(n / 14.1 + (n / 126) ** 2 - (n / 23.6) ** 3 + (n / 25.4) ** 4 - 2.8)


def crr_cycles(dr, n_cycles, b=0.34):
    """Target CSR causing liquefaction in n uniform cycles: CRR15 * (N/15)^-b."""
    return crr_m75(dr) * (n_cycles / 15.0) ** (-b)


def void_ratio(dr):
    return E_MAX - dr * (E_MAX - E_MIN)


def porosity(dr):
    e = void_ratio(dr)
    return e / (1 + e)


def rho_sat(dr):
    """Saturated mass density (Mg/m^3)."""
    e = void_ratio(dr)
    return (GS + e) * RHO_W / (1 + e)


def load_hpo(dr):
    """hpo for this Dr, linearly interpolated from the calibration file (run calibrate.py first)."""
    if not CALIB_FILE.exists():
        raise FileNotFoundError(f"{CALIB_FILE.name} missing - run soil_column/calibrate.py")
    pts = sorted((float(k), v["hpo"]) for k, v in json.loads(CALIB_FILE.read_text()).items())
    if dr <= pts[0][0]:
        return pts[0][1]
    for (d0, h0), (d1, h1) in zip(pts, pts[1:]):
        if dr <= d1:
            return h0 + (h1 - h0) * (dr - d0) / (d1 - d0)
    return pts[-1][1]


def pm4sand_args(dr, hpo=None):
    """Positional arguments for ops.nDMaterial('PM4Sand', tag, *args)."""
    hpo = load_hpo(dr) if hpo is None else hpo
    return [dr, g0_from_dr(dr), hpo, rho_sat(dr), P_ATM]
