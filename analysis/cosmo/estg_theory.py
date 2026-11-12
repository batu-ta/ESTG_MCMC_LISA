"""
Cobaya theory wrapper: hi_class (Python module ``hiclassy``) for ESTG.

Model
-----
The ESTG Jordan-frame action (paper Eq. 3) is

    S = int sqrt(-g) [ F(phi) R / 2 - (d phi)^2 / 2 - V(phi) ],   F = M_P^2 exp(2 beta phi / M_P).

With Phi = F(phi) this is Brans-Dicke, Phi R/2 - omega/(2 Phi) (d Phi)^2 - V, with

    omega_BD(Phi) = F / F'^2 = M_P^2 / (4 beta^2 F)  ~=  1 / (4 beta^2),

since F stays within a few per cent of M_P^2 after BBN. On cosmological scales
the chameleon potential acts as a cosmological constant (a chameleon cannot
drive acceleration differently from Lambda; Wang, Hui & Khoury 2012). So the
cosmological limit of ESTG is Brans-Dicke + Lambda, which hi_class implements
covariantly as ``gravity_model = brans_dicke`` with parameters
(V0, omega_BD, phi_ini, phi_prime_ini). V0 is fixed by the closure relation and
phi_ini by requiring M_*^2 = 1 today (G today = Newton's constant).

We sample ``beta``; alpha_M0 (= running of the Planck mass today) and alpha_B0
are read from hi_class's background and reported as derived parameters.

Why not the alpha-basis parameterisation of the paper (Eq. 15)?
hi_class's ``propto_omega`` with alpha_B = -alpha_M (the scalar-tensor relation;
hi_class uses the Bellini-Sawicki convention, checked numerically) and
alpha_T = 0 has a gradient instability for every c_M != 0: c_s^2 ~ -c_M at
early times for c_M > 0 and c_s^2 ~ +c_M today for c_M < 0. The time dependence
alpha ~ Omega_DE is not one that a consistent Brans-Dicke background produces.

``model: gr`` runs plain LCDM through the same code for the comparison run.

Since February 2026 hi_class ships its wrapper as ``hiclassy.HiClass``;
Cobaya's ``classy`` theory only imports ``classy``, so ``initialize`` loads
``hiclassy`` itself and then repeats the rest of Cobaya's
``classy.initialize`` (Cobaya 3.6.2).
"""

import types

from cobaya.log import LoggedError
from cobaya.theories.classy import classy
from cobaya.theories.classy.classy import (
    non_linear_default_code,
    non_linear_null_value,
)
from cobaya.theories.cosmo import BoltzmannBase

CUSTOM_DERIVED = ("alpha_M0", "alpha_B0", "omega_BD")

# Settings from hi_class's gravity_models/brans_dicke.ini
BRANS_DICKE_SETTINGS = {
    "Omega_Lambda": 0,
    "Omega_fld": 0,
    "Omega_smg": -1,
    "gravity_model": "brans_dicke",
    "M2_tuning_smg": "yes",
    "M2_today_smg": 1.0,
    "pert_initial_conditions_smg": "zero",
    "a_min_stability_test_smg": 1e-6,
    "a_ini_over_a_today_default": 1e-10,
    "a_ini_test_qs_smg": 1e-10,
    "pert_ic_ini_z_ref_smg": 1e9,
}


class HiClassESTG(classy):
    # "brans_dicke" (ESTG) or "gr" (LCDM through the same code)
    model: str = "brans_dicke"
    # omega_BD = 1/(4 max(beta, beta_floor)^2); beta -> 0 is GR
    beta_floor: float = 1e-4

    def initialize(self):
        try:
            import hiclassy
        except ImportError as excpt:
            raise LoggedError(
                self.log, "Could not import 'hiclassy'. Install hi_class with "
                "`pip install ~/estg/hi_class` (see analysis/setup_wsl.sh).") from excpt
        if self.model not in ("brans_dicke", "gr"):
            raise LoggedError(self.log, "model must be 'brans_dicke' or 'gr'")
        # Present hiclassy under the names Cobaya's classy wrapper uses.
        self.classy_module = types.SimpleNamespace(
            Class=hiclassy.HiClass,
            CosmoComputationError=hiclassy.CosmoComputationError,
            CosmoSevereError=hiclassy.CosmoSevereError,
            __version__=getattr(hiclassy, "__version__", None),
        )
        self.classy = hiclassy.HiClass()
        self.log.info("Using hiclassy %s (%s), model=%s",
                      self.classy_module.__version__, hiclassy.__file__, self.model)
        BoltzmannBase.initialize(self)

        # --- same as cobaya.theories.classy.classy.initialize (3.6.2) ---
        self.extra_args["output"] = self.extra_args.get("output", "")
        if "non linear" in self.extra_args:
            if "non_linear" in self.extra_args:
                raise LoggedError(self.log, "In `extra_args`, only one of "
                                  "`non_linear` or `non linear` should be defined.")
            self.extra_args["non_linear"] = self.extra_args.pop("non linear")
        if self.extra_args.get("non_linear", "dummy_string") in (None, False):
            self.extra_args["non_linear"] = non_linear_null_value
        elif "non_linear" not in self.extra_args or self.extra_args["non_linear"] is True:
            self.extra_args["non_linear"] = non_linear_default_code
        self.derived_extra = []

        if self.model == "brans_dicke":
            for key, value in BRANS_DICKE_SETTINGS.items():
                self.extra_args.setdefault(key, value)
        self._omega_bd = float("inf")

    def get_can_provide_params(self):
        return list(super().get_can_provide_params()) + list(CUSTOM_DERIVED)

    def set(self, params_values_dict):
        params = dict(params_values_dict)
        # float(): Cobaya passes np.float64, whose repr is "np.float64(...)" in NumPy 2
        beta = float(params.pop("beta", 0.0))
        if self.model == "brans_dicke":
            self._omega_bd = 1.0 / (4.0 * max(abs(beta), self.beta_floor) ** 2)
            # (V0 [tuned by closure], omega_BD, phi_ini [tuned: M*^2 today = 1], phi'_ini)
            params["parameters_smg"] = f"0.7, {self._omega_bd!r}, 1.0, 0.0"
        return super().set(params)

    def _get_derived_all(self, derived_requested=True):
        # hi_class cannot compute our custom names, so hide them from Cobaya's
        # generic getter (they can sit in both output_params and derived_extra)
        # and fill them in from the background table afterwards.
        custom = [p for p in self.output_params if p in CUSTOM_DERIVED] \
            if derived_requested else []
        custom_extra = [p for p in self.derived_extra if p in CUSTOM_DERIVED]
        saved_out, saved_extra = self.output_params, self.derived_extra
        self.output_params = [p for p in saved_out if p not in CUSTOM_DERIVED]
        self.derived_extra = [p for p in saved_extra if p not in CUSTOM_DERIVED]
        try:
            derived, derived_extra = super()._get_derived_all(derived_requested)
        finally:
            self.output_params, self.derived_extra = saved_out, saved_extra
        if custom or custom_extra:
            if self.model == "brans_dicke":
                bg = self.classy.get_background()
                values = {"alpha_M0": bg["M2_running_smg"][-1],
                          "alpha_B0": bg["braiding_smg"][-1],
                          "omega_BD": self._omega_bd}
            else:
                values = {"alpha_M0": 0.0, "alpha_B0": 0.0, "omega_BD": float("inf")}
            for p in custom:
                derived[p] = float(values[p])
            for p in custom_extra:
                derived_extra[p] = float(values[p])
        return derived, derived_extra
