from __future__ import print_function, division, absolute_import
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from pynflation.potential import InflationaryPotential
from pynflation.parameters import MP


class PlanckSurface(InflationaryPotential):
    def __init__(self, pot_params, num=100, phi_cut=10.):
        InflationaryPotential.__init__(self, pot_params)
        self.phi_root = np.asarray(self.inv_V(MP**4))
        if not np.isfinite(self.phi_root[-1]):
            self.phi_root[-1] = phi_cut
        self.phi = np.concatenate((np.linspace(self.phi_root[+0], 0, num // 2),
                                   np.linspace(0, self.phi_root[-1], num // 2)))
        self.dphi = self.planck_surface()
        self.dphi_SR = self.inflating_region()
        self.dphi_KD = self.KD_region()

    def planck_surface(self):
        dphi2 = 2 * MP**4 - 2 * self.V(self.phi)  # type: np.ndarray
        dphi2 = np.where(np.isclose(dphi2, np.zeros_like(dphi2), atol=1e-3),
                         np.zeros_like(dphi2),
                         dphi2)
        dphi = np.sqrt(dphi2)
        return dphi

    def inflating_region(self):
        dphi2 = self.V(self.phi)
        dphi = np.sqrt(dphi2)
        return dphi

    def KD_region(self):
        dphi2 = 100 * self.V(self.phi)
        dphi = np.sqrt(dphi2)
        return dphi

    def plot_phase_space(self, xmin=None, xmax=None, cSR='0.8', cKD='0.5', cPl='k', alpha=1):
        ax = plt.gca()
        ax.set_xlabel("$\phi(t)~/~\mathrm{m_p}$")
        ax.set_ylabel("$\dot\phi(t)~/~\mathrm{m_p^2}$")
        p_Pl = ax.plot(self.phi, +self.dphi, ':', c=cPl)
        p_Pl = ax.plot(self.phi, -self.dphi, ':', c=cPl)
        p_SR = ax.fill_between(self.phi,
                               -np.minimum(np.abs(self.dphi), np.abs(self.dphi_SR)),
                               +np.minimum(np.abs(self.dphi), np.abs(self.dphi_SR)),
                               color=cSR, alpha=alpha)
        p_KD = ax.fill_between(self.phi,
                               np.abs(self.dphi_KD), np.abs(self.dphi),
                               where=self.dphi_KD < self.dphi, color=cKD, alpha=alpha)
        p_KD = ax.fill_between(self.phi,
                               -np.abs(self.dphi_KD), -np.abs(self.dphi),
                               where=self.dphi_KD < self.dphi, color=cKD, alpha=alpha)
        ax.set_xlim(xmin, xmax)
        return p_Pl[0], p_SR, p_KD


class KDFraction(InflationaryPotential):
    """
    Class to calculate the fraction of the Planck surface that is kinetically dominated
    for different priors:

        frac_uni: uniform in phi
        frac_log: logarithmic in phi
        frac_eng: uniform in the potential (or kinetic) energy distribution
        frac_arc: uniform on the Planck surface (i.e. ratio of arclengths)
    """
    def __init__(self, pot_params, phi_cut=np.inf):
        """
        :param list pot_params: potential parameters of form [<name>, <amplitude array>, <arg>]
        :param phi_cut: upper bound for 'starobinsky' model (default: np.inf)
        """
        InflationaryPotential.__init__(self, pot_params=pot_params)
        assert self.pot_model in ['m2phi2', 'm2phi4', 'm2phi6', 'starobinsky', 'hilltop'], \
            "only designed KDFraction for 'm2phi2', 'm2phi4', 'm2phi6', 'starobinsky', 'hilltop'"
        self.phi_Pl_root = np.asarray(self.inv_V(MP**4))
        self.phi_KD_root = np.asarray(self.inv_V(MP**4 / 51))
        self.phi_Pl_root[-1, ~np.isfinite(self.phi_Pl_root[-1, :])] = phi_cut  # upper bound inf
        self.phi_KD_root[-1, ~np.isfinite(self.phi_KD_root[-1, :])] = phi_cut  # upper bound inf
        self.phi_Pl_min = self._get_extremum(self.phi_Pl_root, -2)  # type: np.ndarray
        self.phi_Pl_max = self._get_extremum(self.phi_Pl_root, -1)  # type: np.ndarray
        self.phi_KD_min = self._get_extremum(self.phi_KD_root, -2)  # type: np.ndarray
        self.phi_KD_max = self._get_extremum(self.phi_KD_root, -1)  # type: np.ndarray
        assert np.all(self.phi_Pl_max > 0)
        assert np.all(self.phi_KD_max > 0)

    @staticmethod
    def _get_extremum(root, idx):
        """
        get extremum from root array
        :param np.ndarray root: array of roots for Planck surface or KD region respectively
        :param int idx: -2 for minimum and -1 for maximum
        :return: extremum
        """
        extremum = np.array([root[-2:, i][idx] for i in range(np.size(root, -1))])
        extremum[np.isnan(extremum)] = 0  # hilltop potential for small amplitudes: start at 0
        return extremum

    def frac_uni(self, use_eff_range=False):
        """
        fraction f of KD regime on Planck surface in phi,phidot phase-space
        for a uniform prior on phi
        :param bool use_eff_range: whether to use full range (False) or an effective range (True)
        :return: float
            fraction f
        """
        phi_Pl_min, phi_Pl_max, phi_KD_min, phi_KD_max = self._phi_ext_eff(use_eff_range)
        l_Pl = phi_Pl_max - phi_Pl_min
        l_KD = phi_KD_max - phi_KD_min
        f_KD = np.where(l_Pl == l_KD, np.ones_like(l_Pl), l_KD / l_Pl)
        return f_KD

    def frac_log(self, c=1e-4, use_eff_range=False):
        """
        fraction f of KD regime on Planck surface in phi,phidot phase-space
        for a logarithmic prior on phi
        :param float c: fraction of phi_max from which to start log prior
        :param bool use_eff_range: whether to use full range (False) or an effective range (True)
        :return: float
            fraction f
        """
        phi_c = c * np.abs(np.where(np.isfinite(self.phi_Pl_max),
                                    self.phi_Pl_max,
                                    self.phi_Pl_min))  # type: np.ndarray
        assert np.all(phi_c > 0)
        phi_Pl_min, phi_Pl_max, phi_KD_min, phi_KD_max = self._phi_ext_eff(use_eff_range)
        phi_Pl_min = np.where(phi_Pl_min == 0, phi_c, phi_Pl_min)
        phi_KD_min = np.where(phi_KD_min == 0, phi_c, phi_KD_min)
        assert np.all(phi_Pl_min != 0) and np.all(phi_KD_min != 0)
        l_Pl = np.where(phi_Pl_min > 0,
                        np.log(phi_Pl_max) - np.log(phi_Pl_min) +
                        np.log(phi_Pl_max) - np.log(phi_Pl_min),
                        np.log(+phi_Pl_max) - np.log(phi_c) +
                        np.log(-phi_Pl_min) - np.log(phi_c))
        l_KD = np.where(phi_KD_min > 0,
                        np.log(phi_KD_max) - np.log(phi_KD_min) +
                        np.log(phi_KD_max) - np.log(phi_KD_min),
                        np.log(+phi_KD_max) - np.log(phi_c) +
                        np.log(-phi_KD_min) - np.log(phi_c))
        f_KD = np.where(l_Pl == l_KD, np.ones_like(l_Pl), l_KD / l_Pl)
        return f_KD

    def frac_eng(self, use_eff_range=False):
        """
        fraction f of KD regime on Planck surface in phi,phidot phase-space
        for a uniform prior on the potential (vs kinetic) energy
        :param bool use_eff_range: whether to use full range (False) or an effective range (True)
        :return: float
            fraction f
        """
        phi_Pl_min, phi_Pl_max, phi_KD_min, phi_KD_max = self._phi_ext_eff(use_eff_range)
        l_Pl = self.V(phi_Pl_max) + self.V(phi_Pl_min)
        l_KD = self.V(phi_KD_max) + self.V(phi_KD_min)
        f_KD = np.where(l_Pl == l_KD, np.ones_like(l_Pl), l_KD / l_Pl)
        return f_KD

    def frac_arc(self, use_eff_range=False):
        """
        fraction f of KD regime on Planck surface in phi,phidot phase-space
        for a uniform prior on the Planck surface (i.e. ratio of arc lengths)
        :param bool use_eff_range: whether to use full range (False) or an effective range (True)
        :return: float
            fraction f
        """
        phi_Pl_min, phi_Pl_max, phi_KD_min, phi_KD_max = self._phi_ext_eff(use_eff_range)
        assert np.all(np.isfinite(phi_Pl_min)) and np.all(np.isfinite(phi_KD_min))
        l_Pl = np.where(np.isfinite(phi_Pl_max),
                        self._integrate_arc_ND(phi_Pl_min, phi_Pl_max),
                        np.inf)
        l_KD = np.where(np.isfinite(phi_KD_max),
                        self._integrate_arc_ND(phi_KD_min, phi_KD_max),
                        np.inf)
        f_KD = np.where(l_Pl == l_KD, np.ones_like(l_Pl), l_KD / l_Pl)
        return f_KD

    def _phi_ext_eff(self, use_eff_range):
        """
        update phi minima and maxima for an effective range of interest for certain potentials
        :param bool use_eff_range: whether to use full range (False) or an effective range (True)
        :return: phi_Pl_min, phi_Pl_max, phi_KD_min, phi_KD_max
        """
        phi_Pl_max = self.phi_Pl_max
        phi_Pl_min = self.phi_Pl_min
        phi_KD_max = self.phi_KD_max
        phi_KD_min = self.phi_KD_min
        if use_eff_range:
            if self.pot_model == 'starobinsky':
                phi_Pl_min = np.zeros_like(self.phi_Pl_min)  # type: np.ndarray
                phi_KD_min = np.zeros_like(self.phi_KD_min)  # type: np.ndarray
            elif self.pot_model == 'hilltop':
                phi_Pl_max = np.ones_like(self.phi_Pl_max) * self.mu  # type: np.ndarray
                phi_KD_max = np.ones_like(self.phi_KD_max) * self.mu  # type: np.ndarray
        return phi_Pl_min, phi_Pl_max, phi_KD_min, phi_KD_max

    def _integrate_arc_ND(self, phi_min, phi_max):
        """
        arc lengths for an array of potential amplitudes
        :param np.ndarray phi_min: array of lower integration limits
        :param np.ndarray phi_max: array of upper integration limits
        :return: array of arc lengths
        """
        return np.array([self._integrate_arc_1D(self._pot_params_i(self.pot_params[:], i),
                                                phi_min.item(i), phi_max.item(i))
                         for i in range(np.size(self.phi_Pl_root, 1))])

    @staticmethod
    def _integrate_arc_1D(pot_params, phi_min, phi_max):
        """
        arc length for a single pot_params instance from phi_min to phi_max
        :param list pot_params: potential parameter instance with only one amplitude element
        :param float phi_min: lower integration limit
        :param float phi_max: upper integration limit
        :return: arc length
        """
        Pot = InflationaryPotential(pot_params)
        arc = integrate.quad(lambda x: np.sqrt(1 + Pot.dVdphi(x)**2 / (2 * MP**4 - 2 * Pot.V(x))),
                             a=phi_min, b=phi_max, full_output=False)
        return arc[0]

    @staticmethod
    def _pot_params_i(pot_params, i):
        """
        cycles through potential amplitude of pot_params
        :param list pot_params: potential parameters where the amplitude pot_params[1] is an array
        :param int i: index for amplitude array
        :return: pot_params instance where the amplitude array got reduced to element i
        """
        pot_params[1] = pot_params[1][i]
        return pot_params


# ******************
# ***   m2phi2   ***
# ******************

def frac_uni_m2phi2():
    """
    fraction f of KD regime on Planck surface in phi,phidot phase-space for Quadratic potential
    for a uniform prior on phi
    :return: float
        fraction f
    """
    # l_KD = 2 * MP**2 / m / np.sqrt(51)
    # l_Pl = 2 * MP**2 / m
    return 1 / np.sqrt(51)


def frac_log_m2phi2(c=1e-4):
    """
    fraction f of KD regime on Planck surface in phi,phidot phase-space for Quadratic potential
    for a logarithmic prior on phi
    :param float c: fraction of phi_max from which to start log prior
    :return: float
        fraction f
    """
    # if mass is not None:
    #     # optional calculation of c as the minimum to produce enough e-folds:
    #     try:
    #         phi_c = efolds2phi_planck(N_tot=60, phi_min=1, phi_max=30, var_type='time',
    #                                   var=np.logspace(0, 2 - np.log10(mass), 3000),
    #                                   pot_params=['m2phi2', mass], verbose=False)[1]
    #         c = min(1 - 1e-15, phi_c * mass / MP**2)
    #     except:
    #         return np.nan
    # # l_KD = np.log(MP**2 / mass / np.sqrt(51)) - np.log(c * MP**2 / mass)
    # # l_Pl = np.log(MP**2 / mass)               - np.log(c * MP**2 / mass)
    return max(0, np.log(c * np.sqrt(51)) / np.log(c))


def frac_eng_m2phi2():
    """
    fraction f of KD regime on Planck surface in phi,phidot phase-space for Quadratic potential
    for a uniform prior on the potential (vs kinetic) energy
    :return: float
        fraction f
    """
    # l_KD = MP**4 / m**2 / 51
    # l_Pl = MP**4 / m**2
    return 1. / 51


def frac_arc_m2phi2(mass):
    """
    fraction f of KD regime on Planck surface in phi,phidot phase-space for Quadratic potential
    for a uniform prior on the Planck surface (i.e. ratio of arc lengths)
    :param float mass: inflaton mass
    :return: float
        fraction f
    """
    Pot = InflationaryPotential(pot_params=['m2phi2', mass])
    l_Pl = integrate.quad(lambda x: np.sqrt(1 + Pot.dVdphi(x)**2 / (2 * MP**4 - 2 * Pot.V(x))),
                          a=Pot.inv_V(MP**4)[0], b=Pot.inv_V(MP**4)[-1])
    l_KD = integrate.quad(lambda x: np.sqrt(1 + Pot.dVdphi(x)**2 / (2 * MP**4 - 2 * Pot.V(x))),
                          a=Pot.inv_V(MP**4 / 51)[0], b=Pot.inv_V(MP**4 / 51)[-1])
    return l_KD[0] / l_Pl[0]


# ***********************
# ***   Starobinsky   ***
# ***********************

def frac_uni_starob(lam2):
    """
    fraction f of KD regime on Planck surface in phi,phidot phase-space for Starobinsky potential
    for a uniform prior on phi
    :param float lam2: squared amplitude of potential
    :return: float
        fraction f
    """
    if lam2 > MP**2:
        l_Pl = -np.log(1 - MP**2 / lam2) + np.log(1 + MP**2 / lam2)
        l_KD = -np.log(1 - MP**2 / lam2 / np.sqrt(51)) + np.log(1 + MP**2 / lam2 / np.sqrt(51))
        assert l_Pl > 0 and l_KD > 0
        return l_KD / l_Pl
    elif lam2 > MP**2 / np.sqrt(51):
        return 0
    else:
        return 1


def frac_uni_starob_eff(lam2):
    """
    fraction f of KD regime on Planck surface in phi,phidot phase-space for Starobinsky potential
    for a uniform prior on phi, where phi is restricted to an effective range of phi > 0
    :param float lam2: squared amplitude of potential
    :return: float
        fraction f
    """
    if lam2 > MP**2:
        l_Pl = -np.log(1 - MP**2 / lam2)
        l_KD = -np.log(1 - MP**2 / lam2 / np.sqrt(51))
        assert l_Pl > 0 and l_KD > 0
        return l_KD / l_Pl
    elif lam2 > MP**2 / np.sqrt(51):
        return 0
    else:
        return 1


def frac_log_starob(lam2, c=1e-4):
    """
    fraction f of KD regime on Planck surface in phi,phidot phase-space for Starobinsky potential
    for a logarithmic prior on phi
    :param float lam2: squared amplitude of potential
    :param float c: fraction of phi_min and phi_max from which to start log prior
    :return: float
        fraction f
    """
    if lam2 > MP**2:
        # try:
        #     phi_c = efolds2phi_planck(N_tot=60, phi_min=5, phi_max=20, var_type='time',
        #                               var=np.logspace(0, 3 - np.log10(lam2), 3000),
        #                               pot_params=['starobinsky', lam2], verbose=False)[1]
        # except:
        #     return np.nan
        # phi_c1 = - c * np.sqrt(3/2) * np.log(1 - MP**2 / lam2)
        phi_c = + c * np.sqrt(3/2) * np.log(1 + MP**2 / lam2)
        # assert phi_c1 > 0
        assert phi_c > 0
        l_Pl1 = np.log(- np.sqrt(3/2) * np.log(1 - MP**2 / lam2)) - np.log(phi_c)
        l_Pl2 = np.log(+ np.sqrt(3/2) * np.log(1 + MP**2 / lam2)) - np.log(phi_c)
        l_KD1 = np.log(- np.sqrt(3/2) * np.log(1 - MP**2 / lam2 / np.sqrt(51))) - np.log(phi_c)
        l_KD2 = np.log(+ np.sqrt(3/2) * np.log(1 + MP**2 / lam2 / np.sqrt(51))) - np.log(phi_c)
        assert l_Pl1 > 0 and l_Pl2 > 0 and l_KD1 > 0 and l_KD2 > 0
        return (l_KD1 + l_KD2) / (l_Pl1 + l_Pl2)
    elif lam2 > MP**2 / np.sqrt(51):
        return 0
    else:
        return 1


def frac_log_starob_eff(lam2, c=1e-4):
    """
    fraction f of KD regime on Planck surface in phi,phidot phase-space for Starobinsky potential
    for a logarithmic prior on phi, where phi is restricted to an effective range of phi > 0
    :param float lam2: squared amplitude of potential
    :param float c: fraction of phi_max from which to start log prior
    :return: float
        fraction f
    """
    if lam2 > MP**2:
        phi_c = + c * np.sqrt(3/2) * np.log(1 + MP**2 / lam2)
        assert phi_c > 0
        l_Pl1 = np.log(- np.sqrt(3/2) * np.log(1 - MP**2 / lam2)) - np.log(phi_c)
        l_KD1 = np.log(- np.sqrt(3/2) * np.log(1 - MP**2 / lam2 / np.sqrt(51))) - np.log(phi_c)
        return l_KD1 / l_Pl1
    elif lam2 > MP**2 / np.sqrt(51):
        return 0
    else:
        return 1


def frac_eng_starob(lam2):
    """
    fraction f of KD regime on Planck surface in phi,phidot phase-space for Starobinsky potential
    for a uniform prior on the potential (vs kinetic) energy
    :param float lam2: squared amplitude of potential
    :return: float
        fraction f
    """
    Pot = InflationaryPotential(pot_params=['starobinsky', lam2])
    phi_Pl_min = - np.sqrt(3/2) * np.log(1 + MP**2 / lam2)
    phi_KD_min = - np.sqrt(3/2) * np.log(1 + MP**2 / lam2 / np.sqrt(51))
    if lam2 <= MP**2 / np.sqrt(51):
        return (lam2**2 + Pot.V(phi_KD_min)) / (lam2**2 + Pot.V(phi_Pl_min))
    else:
        phi_KD_max = - np.sqrt(3/2) * np.log(1 - MP**2 / lam2 / np.sqrt(51))
    if lam2 <= MP**2:
        return (Pot.V(phi_KD_max) + Pot.V(phi_KD_min)) / (lam2**2 + Pot.V(phi_Pl_min))
    else:
        phi_Pl_max = - np.sqrt(3/2) * np.log(1 - MP**2 / lam2)
        return (Pot.V(phi_KD_max) + Pot.V(phi_KD_min)) / (Pot.V(phi_Pl_max) + Pot.V(phi_Pl_min))


def frac_eng_starob_eff(lam2):
    """
    fraction f of KD regime on Planck surface in phi,phidot phase-space for Starobinsky potential
    for a uniform prior on the potential (vs kinetic) energy,
    where phi is restricted to an effective range of phi > 0
    :param float lam2: squared amplitude of potential
    :return: float
        fraction f
    """
    Pot = InflationaryPotential(pot_params=['starobinsky', lam2])
    if lam2 <= MP**2 / np.sqrt(51):
        return 1
    else:
        phi_KD_max = - np.sqrt(3/2) * np.log(1 - MP**2 / lam2 / np.sqrt(51))
    if lam2 <= MP**2:
        return (Pot.V(phi_KD_max) + 0) / (lam2**2 + 0)
    else:
        phi_Pl_max = - np.sqrt(3/2) * np.log(1 - MP**2 / lam2)
        return (Pot.V(phi_KD_max) + 0) / (Pot.V(phi_Pl_max) + 0)


def frac_arc_starob(lam2):
    """
    fraction f of KD regime on Planck surface in phi,phidot phase-space for Starobinsky potential
    for a uniform prior on the Planck surface (i.e. ratio of arc lengths)
    :param float lam2: squared amplitude of potential
    :return: float
        fraction f
    """
    Pot = InflationaryPotential(pot_params=['starobinsky', lam2])
    if lam2 <= MP**2 / np.sqrt(51):
        return 1
    elif lam2 <= MP**2:
        return 0
    else:
        l_Pl = integrate.quad(lambda x: np.sqrt(1 + Pot.dVdphi(x)**2 / (2 * MP**4 - 2 * Pot.V(x))),
                              a=Pot.inv_V(MP**4)[0], b=Pot.inv_V(MP**4)[-1])
        l_KD = integrate.quad(lambda x: np.sqrt(1 + Pot.dVdphi(x)**2 / (2 * MP**4 - 2 * Pot.V(x))),
                              a=Pot.inv_V(MP**4 / 51)[0], b=Pot.inv_V(MP**4 / 51)[-1])
        return l_KD[0] / l_Pl[0]


def frac_arc_starob_eff(lam2):
    """
    fraction f of KD regime on Planck surface in phi,phidot phase-space for Starobinsky potential
    for a uniform prior on the Planck surface (i.e. ratio of arc lengths),
    where phi is restricted to an effective range of phi > 0
    :param float lam2: squared amplitude of potential
    :return: float
        fraction f
    """
    Pot = InflationaryPotential(pot_params=['starobinsky', lam2])
    if lam2 <= MP**2 / np.sqrt(51):
        return 1
    elif lam2 <= MP**2:
        return 0
    else:
        l_Pl = integrate.quad(lambda x: np.sqrt(1 + Pot.dVdphi(x)**2 / (2 * MP**4 - 2 * Pot.V(x))),
                              a=0, b=Pot.inv_V(MP**4)[-1])
        l_KD = integrate.quad(lambda x: np.sqrt(1 + Pot.dVdphi(x)**2 / (2 * MP**4 - 2 * Pot.V(x))),
                              a=0, b=Pot.inv_V(MP**4 / 51)[-1])
        return l_KD[0] / l_Pl[0]


# *******************
# ***   Hilltop   ***
# *******************

def frac_uni_hilltp(lam2):
    """
    fraction f of KD regime on Planck surface in phi,phidot phase-space for Hilltop potential
    for a uniform prior on phi
    :param float lam2: squared amplitude of potential
    :return: float
        fraction f
    """
    phi_Pl_max = np.sqrt(1 + MP**2 / lam2)
    phi_KD_max = np.sqrt(1 + MP**2 / lam2 / np.sqrt(51))
    assert phi_Pl_max > 0 and phi_KD_max > 0
    if lam2 <= MP**2 / np.sqrt(51):
        return phi_KD_max / phi_Pl_max
    else:
        phi_KD_min = np.sqrt(1 - MP**2 / lam2 / np.sqrt(51))
        assert phi_KD_min > 0
    if lam2 <= MP**2:
        return (phi_KD_max - phi_KD_min) / phi_Pl_max
    else:
        phi_Pl_min = np.sqrt(1 - MP**2 / lam2)
        assert phi_Pl_min > 0
        return (phi_KD_max - phi_KD_min) / \
               (phi_Pl_max - phi_Pl_min)


def frac_uni_hilltp_eff(lam2):
    """
    fraction f of KD regime on Planck surface in phi,phidot phase-space for Hilltop potential
    for a uniform prior on phi, where phi is restricted to an effective range of mu < phi < mu
    :param float lam2: squared amplitude of potential
    :return: float
        fraction f
    """
    if lam2 <= MP**2 / np.sqrt(51):
        return 1
    else:
        phi_KD_min = np.sqrt(1 - MP**2 / lam2 / np.sqrt(51))
        assert phi_KD_min > 0
    if lam2 <= MP**2:
        return 1 - phi_KD_min
    else:
        phi_Pl_min = np.sqrt(1 - MP**2 / lam2)
        assert phi_Pl_min > 0
        return (1 - phi_KD_min) / (1 - phi_Pl_min)


def frac_log_hilltp(lam2, c=1e-4):
    """
    fraction f of KD regime on Planck surface in phi,phidot phase-space for Hilltop potential
    for a uniform prior on phi
    :param float lam2: squared amplitude of potential
    :param float c: fraction of phi_max from which to start log prior
    :return: float
        fraction f
    """
    phi_Pl_max = np.sqrt(1 + MP**2 / lam2)
    phi_KD_max = np.sqrt(1 + MP**2 / lam2 / np.sqrt(51))
    phi_c = c * phi_Pl_max
    assert phi_Pl_max > 0 and phi_KD_max > 0
    if lam2 <= MP**2 / np.sqrt(51):
        return (np.log(phi_KD_max) - np.log(phi_c)) / \
               (np.log(phi_Pl_max) - np.log(phi_c))
    else:
        phi_KD_min = np.sqrt(1 - MP**2 / lam2 / np.sqrt(51))
        assert phi_KD_min > 0
    if lam2 <= MP**2:
        return (np.log(phi_KD_max) - np.log(phi_KD_min)) / \
               (np.log(phi_Pl_max) - np.log(phi_c))
    else:
        phi_Pl_min = np.sqrt(1 - MP**2 / lam2)
        assert phi_Pl_min > 0
        return (np.log(phi_KD_max) - np.log(phi_KD_min)) / \
               (np.log(phi_Pl_max) - np.log(phi_Pl_min))


def frac_log_hilltp_eff(lam2, c=1e-4):
    """
    fraction f of KD regime on Planck surface in phi,phidot phase-space for Hilltop potential
    for a uniform prior on phi, where phi is restricted to an effective range of mu < phi < mu
    :param float lam2: squared amplitude of potential
    :param float c: fraction of phi_max from which to start log prior
    :return: float
        fraction f
    """
    phi_Pl_max = np.sqrt(1 + MP**2 / lam2)
    phi_c = c * phi_Pl_max
    assert phi_Pl_max > 0
    if lam2 <= MP**2 / np.sqrt(51):
        return 1
    else:
        phi_KD_min = np.sqrt(1 - MP**2 / lam2 / np.sqrt(51))
        assert phi_KD_min > 0
    if lam2 <= MP**2:
        return np.log(phi_KD_min) / np.log(phi_c)
    else:
        phi_Pl_min = np.sqrt(1 - MP**2 / lam2)
        assert phi_Pl_min > 0
        return np.log(phi_KD_min) / np.log(phi_Pl_min)


def frac_eng_hilltp(lam2):
    """
    fraction f of KD regime on Planck surface in phi,phidot phase-space for Hilltop potential
    for a uniform prior on the potential (vs kinetic) energy
    :param float lam2: squared amplitude of potential
    :return: float
        fraction f
    """
    def V(x): return lam2**2 * (1 - x**2)**2
    phi_Pl_max = np.sqrt(1 + MP**2 / lam2)
    phi_KD_max = np.sqrt(1 + MP**2 / lam2 / np.sqrt(51))
    assert phi_Pl_max > 0 and phi_KD_max > 0
    if lam2 <= MP**2 / np.sqrt(51):
        return (lam2**2 + V(phi_KD_max)) / \
               (lam2**2 + V(phi_Pl_max))
    else:
        phi_KD_min = np.sqrt(1 - MP**2 / lam2 / np.sqrt(51))
        assert phi_KD_min > 0
    if lam2 <= MP**2:
        return (V(phi_KD_min) + V(phi_KD_max)) / \
               (lam2**2 + V(phi_Pl_max))
    else:
        phi_Pl_min = np.sqrt(1 - MP**2 / lam2)
        assert phi_Pl_min > 0
        return (V(phi_KD_min) + V(phi_KD_max)) / \
               (V(phi_Pl_min) + V(phi_Pl_max))


def frac_eng_hilltp_eff(lam2):
    """
    fraction f of KD regime on Planck surface in phi,phidot phase-space for Hilltop potential
    for a uniform prior on the potential (vs kinetic) energy,
    where phi is restricted to an effective range of mu < phi < mu
    :param float lam2: squared amplitude of potential
    :return: float
        fraction f
    """
    def V(x): return lam2**2 * (1 - x**2)**2
    if lam2 <= MP**2 / np.sqrt(51):
        return 1
    else:
        phi_KD_min = np.sqrt(1 - MP**2 / lam2 / np.sqrt(51))
        assert phi_KD_min > 0
    if lam2 <= MP**2:
        return V(phi_KD_min) / lam2**2
    else:
        phi_Pl_min = np.sqrt(1 - MP**2 / lam2)
        assert phi_Pl_min > 0
        return V(phi_KD_min) / V(phi_Pl_min)


def frac_arc_hilltp(lam2, mu=1.):
    """
    fraction f of KD regime on Planck surface in phi,phidot phase-space for Hilltop potential
    for a uniform prior on the Planck surface (i.e. ratio of arc lengths)
    :param float lam2: squared amplitude of potential
    :param float mu: potential minimum position
    :return: float
        fraction f
    """
    Pot = InflationaryPotential(pot_params=['hilltop', lam2, mu])
    if lam2 <= MP**2:
        l_Pl = integrate.quad(lambda x: np.sqrt(1 + Pot.dVdphi(x)**2 / (2 * MP**4 - 2 * Pot.V(x))),
                              a=0, b=Pot.inv_V(MP**4)[-1])[0]
    else:
        l_Pl = integrate.quad(lambda x: np.sqrt(1 + Pot.dVdphi(x)**2 / (2 * MP**4 - 2 * Pot.V(x))),
                              a=Pot.inv_V(MP**4)[-2], b=Pot.inv_V(MP**4)[-1])[0]
    if lam2 <= MP**2 / np.sqrt(51):
        l_KD = integrate.quad(lambda x: np.sqrt(1 + Pot.dVdphi(x)**2 / (2 * MP**4 - 2 * Pot.V(x))),
                              a=0, b=Pot.inv_V(MP**4 / 51)[-1])[0]
    else:
        l_KD = integrate.quad(lambda x: np.sqrt(1 + Pot.dVdphi(x)**2 / (2 * MP**4 - 2 * Pot.V(x))),
                              a=Pot.inv_V(MP**4 / 51)[-2], b=Pot.inv_V(MP**4 / 51)[-1])[0]
    return l_KD / l_Pl


def frac_arc_hilltp_eff(lam2, mu=1.):
    """
    fraction f of KD regime on Planck surface in phi,phidot phase-space for Hilltop potential
    for a uniform prior on the Planck surface (i.e. ratio of arc lengths)
    where phi is restricted to an effective range of mu < phi < mu
    :param float lam2: squared amplitude of potential
    :param float mu: potential minimum position
    :return: float
        fraction f
    """
    Pot = InflationaryPotential(pot_params=['hilltop', lam2, mu])
    if lam2 <= MP**2:
        l_Pl = integrate.quad(lambda x: np.sqrt(1 + Pot.dVdphi(x)**2 / (2 * MP**4 - 2 * Pot.V(x))),
                              a=0, b=mu)[0]
    else:
        l_Pl = integrate.quad(lambda x: np.sqrt(1 + Pot.dVdphi(x)**2 / (2 * MP**4 - 2 * Pot.V(x))),
                              a=Pot.inv_V(MP**4)[-2], b=mu)[0]
    if lam2 <= MP**2 / np.sqrt(51):
        l_KD = integrate.quad(lambda x: np.sqrt(1 + Pot.dVdphi(x)**2 / (2 * MP**4 - 2 * Pot.V(x))),
                              a=0, b=mu)[0]
    else:
        l_KD = integrate.quad(lambda x: np.sqrt(1 + Pot.dVdphi(x)**2 / (2 * MP**4 - 2 * Pot.V(x))),
                              a=Pot.inv_V(MP**4 / 51)[-2], b=mu)[0]
    return l_KD / l_Pl
