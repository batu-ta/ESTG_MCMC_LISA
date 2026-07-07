import numpy as np
from montepython.likelihood_class import Likelihood

class estg_likelihood(Likelihood):
    def __init__(self, path, data, command_line):
        Likelihood.__init__(self, path, data, command_line)

    def loglkl(self, cosmo, data):
        chi2_planck = data.planck.get_chi2(cosmo)
        omega_b_h2 = cosmo.Omega_b() * cosmo.h()**2
        chi2_bbn = ((omega_b_h2 - 0.0222) / 0.0005)**2
        alpha_M = cosmo.alpha_M
        chi2_ligo = (alpha_M / 0.05)**2
        return -0.5 * (chi2_planck + chi2_bbn + chi2_ligo)
