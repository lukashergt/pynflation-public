from __future__ import print_function, division, absolute_import
import warnings
import numpy as np
from scipy import optimize, special
from pynflation.parameters import MP
from pynflation.potential import InflationaryPotential
from pynflation.background import InflatonSolver


class SlowRollIC(InflationaryPotential):
    """
    Class for self-consistent initial conditions using the slow-roll approximation.

    Methods
    -------
        initial_conditions_t : set up i.c. for phi, dphidt, H, and a
        initial_conditions_loga : set up i.c. for phi, dphidloga, H, and a

    """
    def __init__(self, pot_params):
        """
        :param pot_params: list
            insert pot_params used by the potential (e.g. mass m)
            where pot_params[0] is a string specifying the potential model
            'm2phi2' : V(phi) = m^2 phi^2
            'm2phi4' : V(phi) = m^2 phi^4
            'm2phi6' : V(phi) = m^2 phi^6
        """
        InflationaryPotential.__init__(self, pot_params)

    def initial_conditions(self, var_type, a_init=1., phi_init=None, N_end=None, phi_end=0.):
        """
        Set up self-consistent initial conditions using the slow-roll approximation.
        Independent variable specified by var_type.

        :param var_type: str
            specifies the independent variable:
            either 'time' for physical time
            or 'loga' for the log of the scale factor
        :param a_init: float
            initial value for scale factor a
        :param float phi_init: initial value for inflaton field phi
        :param N_end: float
            projected number of efolds N from slow roll approximation
            allows to calculate an initial value for phi
        :param phi_end: float
            end value for inflaton field phi for calculation of number of efolds N
        :return: np.ndarray
            array of initial conditions: np.array([a_init, N_init, phi_init, dphi_init])
            Note that dphi_init corresponds to either dphi/dt or dphi/dloga
            depending on the indep. variable
        """
        a_init = a_init
        if var_type == 'time':
            return self.initial_conditions_t(N_end=N_end, phi_end=phi_end, phi_init=phi_init,
                                             a_init=a_init)
        if phi_init is None:
            assert N_end is not None, "Need to specify either phi_init or N_end."
            phi_init = np.sqrt(2. * N_end * self.n * MP**2 + phi_end**2)
        else:
            phi_init = phi_init
        # H_init_t = np.sqrt(self.V(phi_init) / (3. * MP**2))
        # dphi_init_t = - self.dVdphi(phi_init) / (3. * H_init_t)
        if var_type == 'time':
            dphi_init = - MP * self.n * self.m * phi_init**(self.n / 2. - 1) / np.sqrt(3.)
            assert dphi_init ** 2 < self.V(phi_init), \
                "Slow-roll condition dphidt**2 < V(phi) violated."
        elif var_type == 'loga':
            dphi_init = - self.n * MP**2 / phi_init
        else:
            raise Exception("var_type is %s, but only 'time' and 'loga' supported."
                            % var_type)

        assert self.dVV(phi_init)**2/2. < 1, \
            "Slow-roll condition V'(phi)/V(phi)/2 < 1 violated"
        assert self.d2Vdphi2(phi_init)/self.V(phi_init) < 1, \
            "Slow-roll condition V'(phi)/V(phi)/2 < 1 violated"

        return np.array([a_init, phi_init, dphi_init])

    def initial_conditions_t(self, N_end, phi_end, phi_init=None, a_end=None, a_init=None):
        """
        Set up self-consistent initial conditions using the slow-roll approximation.
        Independent variable: t

        :param float N_end: Number of e-folds from start phi_init_t to end phi_end of inflation.
        :param float phi_end: Value of inflaton field phi at the end of inflation.
        :param float phi_init: initial value for inflaton field phi
        :param float a_end: scale factor at the end of inflation
        :param float a_init: initial value for scale factor
        :return:
        """
        if self.pot_model in self._power_law_pot:
            if phi_init is None:
                assert N_end is not None, "Need to specify either phi_init or N_end."
                phi_init_t = np.sqrt(2. * N_end * self.n * MP**2 + phi_end**2)
            else:
                phi_init_t = phi_init

            # H_init_t = np.sqrt(phi_init_t**self.n / 3.) * self.m / MP
            dphi_init_t = - MP * self.n * self.m * phi_init_t**(self.n / 2. - 1) / np.sqrt(3.)
            # dphi_init_t = - self.m * np.sqrt(-phi_init_t**self.n *
            #                                  (1. - np.sqrt(1. + 2. * self.n /
            #                                                3. * MP**2 * phi_init_t**(-2))))
            # phi_init_t = np.sqrt(2. / 3. * special.lambertw(np.sqrt(np.exp(12. * N_end))))
        elif self.pot_model == 'starobinsky':
            if phi_init is None:
                assert N_end is not None, "Need to specify either phi_init or N_end."
                phi_init_t = np.sqrt(3. / 2.) * MP * np.log(4. / 3. * N_end + 1)
            else:
                phi_init_t = phi_init
            H_init_t = np.sqrt(self.V(phi_init_t) / (3. * MP**2))
            dphi_init_t = - self.dVdphi(phi_init_t) / (3. * H_init_t)
        else:
            raise NotImplementedError("pot_model=%s not implemented for SR initial conditions."
                                      % self.pot_model)
        if a_init is None:
            assert a_end is not None, "Either a_end or a_init needs to be specified."
            a_init_t = a_end * np.exp(phi_end / (2. * self.n * MP**2) - N_end)
        else:
            a_init_t = a_init

        assert dphi_init_t**2 < self.V(phi_init_t), \
            "Slow-roll condition dphidt**2 = %e < V(phi) = %e violated." \
            % (dphi_init_t**2, self.V(phi_init_t))
        assert self.dVV(phi_init_t)**2/2. < 1, \
            "Slow-roll condition V'(phi)/V(phi)/2 < 1 violated"
        assert self.d2Vdphi2(phi_init_t)/self.V(phi_init_t) < 1, \
            "Slow-roll condition V''(phi)/V(phi) < 1 violated"

        return np.array([a_init_t, phi_init_t, dphi_init_t])  # , eta_init

    def initial_conditions_a(self, N_end, phi_end, a_end=None, a_init=None):
        """
        Set up self-consistent initial conditions using the slow-roll approximation.
        Independent variable: loga = ln(a)
        suffix _a means w.r.t. loga

        :param N_end: float
            Number of e-folds from start phi_init_a to end phi_end of inflation.
        :param phi_end: float
            Value of inflaton field phi at the end of inflation.
        :param a_end: float
            scale factor at the end of inflation
        :param float a_init:
            initial log of scale factor loga_init = ln(a_init_a)
            by default determined via a_init_a through a_end and N_end
        :return:
        """
        a_init_t, phi_init_t, dphi_init_t = self.initial_conditions_t(N_end, phi_end,
                                                                      a_end, a_init)
        phi_init_a = phi_init_t
        a_init_a = a_init_t
        dphi_init_a = - self.n * MP**2 / phi_init_a

        return np.array([a_init_a, phi_init_a, dphi_init_a])


class KineticDominanceIC(InflationaryPotential):
    """
    Class for the time period of kinetic dominance (KD)
    and for setting up initial conditions during that period.

    Methods
    -------
        initial_conditions   : setting up initial conditions allowing to specify indep. variable
        initial_conditions_t : setting up initial conditions w.r.t. time
        initial_conditions_a : setting up initial conditions w.r.t. loga

        t2a     : conversion function from time t to scale factor a during KD
        t2eta   : conversion function from time t to conformal time eta during KD
        t2loga : conversion function from time t to loga=ln(a) during KD

        a2t     : conversion function from scale factor a to time t during KD
        a2eta   : conversion function from scale factor a to conformal time eta during KD
        a2loga : conversion function from scale factor a to loga=ln(a) during KD

        eta2t : conversion function from conformal time eta to time t during KD
        eta2a : conversion function from conformal time eta to scale factor a during KD

        loga2t : conversion function from loga=ln(a) to time t during KD
        loga2a : conversion function from loga=ln(a) to scale factor a during KD
    """
    def __init__(self, pot_params, t_p=1., a_p=1.):
        """
        :param pot_params: list
            insert pot_params used by the potential (e.g. mass m)
            where pot_params[0] is a string specifying the potential model
            'm2phi2' : V(phi) = m^2 phi^2
            'm2phi4' : V(phi) = m^2 phi^4
            'm2phi6' : V(phi) = m^2 phi^6
        :param t_p: float
            Specifies the time t_p, default 1
        :param a_p: float
            Specifies the scale factor a_p = a(t_p) at time t_p, default 1
        """
        InflationaryPotential.__init__(self, pot_params)
        self.t_p = t_p
        self.a_p = a_p
        self.eta_p = (3. * self.t_p) / (2. * self.a_p)
        self.loga_p = np.log(self.a_p)

    def initial_conditions(self, var_type, var_init, phi_p, verbose=False):
        """
        Set up kinetic dominance (KD) initial conditions.
        Independent variable specified by var_type.

        :param var_type: str
            Choice of independent variable, either 'time' or 'loga'.
        :param var_init: float
            starting time/loga (needs to be chosen sufficiently small for KD)
        :param phi_p: float
            phi_0 as in Handley et al. "Kinetic initial conditions for inflation"
        :param verbose: bool
            if True, prints out initial conditions
        :return: np.ndarray
            array of initial conditions: np.array([a_init, phi_init, dphi_init])
            Note that dphi_init corresponds to either dphi/dt or dphi/dloga
            depending on the indep. variable
        """
        if var_type == 'time':
            ic = self.initial_conditions_t(t_init=var_init, phi_p=phi_p, verbose=verbose)
        elif var_type == 'loga':
            ic = self.initial_conditions_a(loga_init=var_init, phi_p=phi_p)
        else:
            raise Exception("var_type is %s, but only 'time' and 'loga' supported."
                            % var_type)
        return ic

    def initial_conditions_t(self, t_init, phi_p, verbose=False):
        """
        Set up kinetic dominance (KD) initial conditions.
        Independent variable: t

        :param t_init: float
            starting time (needs to be chosen sufficiently small for KD)
        :param phi_p: float
            phi_0 as in Handley et al. "Kinetic initial conditions for inflation"
        :param verbose: bool
            if True, prints out initial conditions
        :return: np.ndarray
            array of initial conditions: np.array([a_init, phi_init, dphi_init])
            Note that dphi_init corresponds to dphi/dt
        """
        verboseprint = print if verbose else lambda *a, **k: None
        a_init_t = self.a_p * (t_init / self.t_p)**(1./3.)
        # H_init_t = 1. / (3. * t_init)
        phi_init_t = phi_p - np.sqrt(2./3.) * MP * np.log(t_init / self.t_p)
        dphi_init_t = - np.sqrt(2./3.) * MP / t_init
        # dphi_init_t = - np.sqrt(6. * MP**2 * H_init_t**2 - 2. * Pot.V(phi_init_t))
        ic = np.array([a_init_t, phi_init_t, dphi_init_t])
        verboseprint("Initial conditions: ")
        verboseprint("a_init = %f  phi_init = %f  dphi_init = %e" % tuple(ic))
        return ic

    def initial_conditions_a(self, loga_init, phi_p):
        """
        Set up kinetic dominance initial conditions.
        Independent variable: loga = ln(a)

        :param loga_init: float
            starting loga
        :param phi_p: float
            phi_0 as in Handley et al. "Kinetic initial conditions for inflation"
        :return: np.ndarray
            array of initial conditions: np.array([a_init, phi_init, dphi_init])
            Note that dphi_init corresponds to dphi/dloga
        """
        loga_p = np.log(self.a_p)
        phi_init_a = phi_p - np.sqrt(6.) * MP * (loga_init - loga_p)
        H_init_a = np.exp(-3. * (loga_init - loga_p)) / (3. * self.t_p)
        dphi_init_a = - np.sqrt(6. * MP**2 - 2. * self.V(phi_init_a) / H_init_a**2)
        a_init_a = np.exp(loga_init)
        return np.array([a_init_a, phi_init_a, dphi_init_a])

    def t2a(self, t):
        return self.a_p * (t / self.t_p) ** (1. / 3.)

    def t2eta(self, t):
        return self.eta_p * (t / self.t_p) ** (2. / 3.)

    def t2loga(self, t):
        return np.log(self.a_p) + np.log(t / self.t_p) / 3.

    def a2t(self, a):
        return self.t_p * (a / self.a_p) ** 3

    def a2eta(self, a):
        return self.eta_p * (a / self.a_p) ** 2

    @staticmethod
    def a2loga(a):
        return np.log(a)

    def eta2t(self, eta):
        return self.t_p * (eta / self.eta_p) ** (3. / 2.)

    def eta2a(self, eta):
        return self.a_p * (eta / self.eta_p) ** (1. / 2.)

    def loga2t(self, loga):
        return self.t_p * np.exp(3. * loga) / self.a_p ** 3

    @staticmethod
    def loga2a(loga):
        return np.exp(loga)


def phi2efolds_KD(phi_p, var_type, var, pot_params, integrator='lsoda'):
    """
    Find total number of e-folds of inflation given phi_0
    :param phi_p: float
        phi_0 as in Handley et al. "Kinetic initial conditions for inflation"
    :param var_type: str
        Choice of independent variable, either 'time' or 'loga'.
    :param var: arraylike
        array of independent variable
    :param pot_params: list
        insert pot_params used by the potential
        where pot_params[0] is a string specifying the potential model
        'm2phi2' : V(phi) = m^2 phi^2
        'm2phi4' : V(phi) = m^2 phi^4
        'm2phi6' : V(phi) = m^2 phi^6
    :param integrator: str
        Choice of integrator, either of 'lsoda' (default), 'vode', 'zvode', 'dopri5', 'dop853'.
    :return: float
        N_tot = total number of e-folds of inflation
    """
    KD = KineticDominanceIC(pot_params=pot_params)
    ic = KD.initial_conditions(var_type=var_type, var_init=var[0], phi_p=phi_p)
    bkd = InflatonSolver(var_type=var_type, var=var, pot_params=pot_params, ic=ic,
                         N_star=0, integrator=integrator, verbose=False)
    if np.nanmax(np.abs(bkd.phi[-10:])) > 1.:
        print("phi might still be inflating, please increase time")
    return bkd.N_tot


def phi2efolds_SR(phi_init, var_type, var, pot_params, a_init=1., phi_end=0., integrator='lsoda'):
    """
    Find total number of e-folds of inflation given phi_0
    :param float phi_init: phi_0 as in Handley et al. "Kinetic initial conditions for inflation"
    :param str var_type: Choice of independent variable, either 'time' or 'loga'.
    :param np.ndarray var: array of independent variable
    :param list pot_params:
        insert pot_params used by the potential
        where pot_params[0] is a string specifying the potential model
        'm2phi2' : V(phi) = m^2 phi^2
        'm2phi4' : V(phi) = m^2 phi^4
        'm2phi6' : V(phi) = m^2 phi^6
        'starobinsky' : V(phi) = L^4 (1 - exp(-sqrt(2/3)*phi))^2
        'hilltop' : V(phi) = L^4 (1 - (phi / mu)^2)^2
    :param float a_init: initial value for scale factor
    :param float phi_end: Value of inflaton field phi at the end of inflation.
    :param str integrator: Choice of integrator,
        either of 'lsoda' (default), 'vode', 'zvode', 'dopri5', 'dop853'.
    :return: float
        N_tot = total number of e-folds of inflation
    """
    SR = SlowRollIC(pot_params=pot_params)
    ic = SR.initial_conditions(var_type=var_type, phi_init=phi_init,
                               a_init=a_init, phi_end=phi_end)
    b = InflatonSolver(var_type=var_type, var=var, pot_params=pot_params, ic=ic, N_star=0,
                       integrator=integrator, verbose=False)
    return b.N_tot


def efolds2phi_KD(N_tot, phi_min, phi_max, var_type, var, pot_params, N_star=0,
                  integrator='lsoda', verbose=True):
    """
    Given the desired total number of e-folds of inflation N_tot,
    find the required initial conditions.
    :param N_tot: float
        total number of e-folds of inflation
    :param phi_min: float
        initial minimum guess for phi_0
    :param phi_max: float
        initial maximum guess for phi_0
    :param var_type: str
        Choice of independent variable, either 'time' or 'loga'.
    :param var: arraylike
        array of independent variable
    :param pot_params: list
        insert pot_params used by the potential
        where pot_params[0] is a string specifying the potential model
        'm2phi2' : V(phi) = m^2 phi^2
        'm2phi4' : V(phi) = m^2 phi^4
        'm2phi6' : V(phi) = m^2 phi^6
    :param float N_star: number of observable e-folds of inflation
        (since not important for determining the total number of e-folds
        it is fine to leave this at its default: 0)
    :param integrator: str
        Choice of integrator, either of 'lsoda' (default), 'vode', 'zvode', 'dopri5', 'dop853'.
    :param verbose: bool
        if True, prints out initial conditions
    :return: np.ndarray
        array of initial conditions: np.array([a_init, phi_init, dphi_init])
    """
    def func(x): return phi2efolds_KD(phi_p=x, var_type=var_type, var=var, pot_params=pot_params,
                                      integrator=integrator) - N_tot
    new_phi_p, output = optimize.brentq(f=func, a=phi_min, b=phi_max, xtol=1e-15, full_output=True)
    KD = KineticDominanceIC(pot_params)
    ic = KD.initial_conditions(var_type=var_type, var_init=var[0], phi_p=new_phi_p)
    if verbose:
        bkd = InflatonSolver(var_type=var_type, var=var, pot_params=pot_params, ic=ic,
                             N_star=N_star, integrator=integrator, verbose=False)
        print(output)
        print("          N_tot: %.15f" % bkd.N_tot)
        print("         N_dagg: %.15f" % bkd.N_dagg)
        print("         N_star: %.15f" % bkd.N_star)
        print()
        print("         a_init: %.15f" % bkd.ic[0])
        print("       phi_init: %.15f" % bkd.ic[1])
        print("      dphi_init: %.15e" % bkd.ic[2])
        print()
    return ic


def efolds2phi_KD_old(N_tot, rtol, atol, var_type, var, pot_params, phi_p=None,
                      integrator='lsoda', verbose=True):
    Pot = InflationaryPotential(pot_params)
    phi_end = 0
    if phi_p is None:
        phi_p = np.sqrt(2. * N_tot * Pot.n * MP**2 + phi_end**2)
    phi_p_max = 3. * phi_p
    phi_p_min = 0.5 * phi_p

    KD = KineticDominanceIC(pot_params)
    ic = KD.initial_conditions(var_type=var_type, var_init=var[0], phi_p=phi_p_max)
    bkd = InflatonSolver(var_type=var_type, var=var, pot_params=pot_params, ic=ic,
                         N_star=0, integrator=integrator, verbose=False)
    assert bkd.N_tot > N_tot, "Initial guess not big enough, please increase phi_0."
    KD = KineticDominanceIC(pot_params)
    ic = KD.initial_conditions(var_type=var_type, var_init=var[0], phi_p=phi_p_min)
    bkd = InflatonSolver(var_type=var_type, var=var, pot_params=pot_params, ic=ic,
                         N_star=0, integrator=integrator, verbose=False)
    assert bkd.N_tot < N_tot, "Initial guess not small enough, please decrease phi_0."

    KD = KineticDominanceIC(pot_params)
    ic = KD.initial_conditions(var_type=var_type, var_init=var[0], phi_p=phi_p)
    bkd = InflatonSolver(var_type=var_type, var=var, pot_params=pot_params, ic=ic,
                         N_star=0, integrator=integrator, verbose=False)

    counter = 0
    counter_max = 100
    while not np.isclose(bkd.N_tot, N_tot, rtol=rtol, atol=atol) and counter < counter_max:
        if bkd.N_tot < N_tot:
            phi_p_min = phi_p
            phi_p = (phi_p_max + phi_p_min) / 2.
        else:
            phi_p_max = phi_p
            phi_p = (phi_p_max + phi_p_min) / 2.
        ic = KD.initial_conditions(var_type=var_type, var_init=var[0], phi_p=phi_p)
        bkd = InflatonSolver(var_type=var_type, var=var, pot_params=pot_params,
                             ic=ic, N_star=0,
                             integrator=integrator, verbose=False)
        counter += 1

    if verbose:
        print("number of runs: ", counter)
        print("Initial conditions used: ")
        print("a_init = %.3f  phi_init = %.3f  dphi_init = %.3e" % tuple(bkd.ic))
        print("Total number of efolds of inflation: N_tot = %f" % bkd.N_tot)
        print("phi_0 = %f" % phi_p)
        print("phi_0 = ")
        print(phi_p)
    if counter >= counter_max:
        warnings.warn("Did not reach desired N_tot!")
    return bkd.ic


def efolds2phi_SR(N_tot, rtol, atol, var_type, var, a0, pot_params,
                  integrator='lsoda', verbose=True):
    Pot = InflationaryPotential(pot_params)
    phi_end = 0

    phi0 = np.sqrt(2. * N_tot * Pot.n * MP**2 + phi_end**2)
    phi0_max = 3. * phi0
    phi0_min = 0.5 * phi0

    SR = SlowRollIC(pot_params=pot_params)
    ic = SR.initial_conditions(var_type=var_type, a_init=a0, phi_init=phi0)
    bsr = InflatonSolver(var_type=var_type, var=var, pot_params=pot_params, ic=ic,
                         N_star=0, integrator=integrator, verbose=False)

    counter = 0
    counter_max = 200
    while not np.isclose(bsr.N_tot, N_tot, rtol=rtol, atol=atol) and counter < counter_max:
        if bsr.N_tot < N_tot:
            phi0_min = phi0
            phi0 = (phi0_max + phi0_min) / 2.
        else:
            phi0_max = phi0
            phi0 = (phi0_max + phi0_min) / 2.
        ic = SR.initial_conditions(var_type=var_type, a_init=a0, phi_init=phi0)
        bsr = InflatonSolver(var_type=var_type, var=var, pot_params=pot_params,
                             ic=ic, N_star=0,
                             integrator=integrator, verbose=False)
        counter += 1

    if verbose:
        print("number of runs: ", counter)
        print("Initial conditions used: ")
        print("a0 = %.3f  phi_init = %.3f  dphi_init = %.3e" % tuple(bsr.ic))
        print("Total number of efolds of inflation: N_tot = %.4f" % bsr.N_tot)
        print("phi0 = %f" % phi0)
    if counter >= counter_max:
        warnings.warn("Did not reach desired N_tot!")
    return bsr.ic


def efolds2phi_SR_new(N_tot, phi_min, phi_max, var_type, var, pot_params, a_init, N_star,
                      integrator='lsoda', verbose=True):
    def func(x): return phi2efolds_SR(phi_init=x, var_type=var_type, var=var, a_init=a_init,
                                      pot_params=pot_params, integrator=integrator) - N_tot
    new_phi_0, output = optimize.brentq(f=func, a=phi_min, b=phi_max, xtol=1e-15, full_output=True)
    SR = SlowRollIC(pot_params=pot_params)
    ic = SR.initial_conditions(var_type=var_type, phi_init=new_phi_0, a_init=a_init)
    if verbose:
        b = InflatonSolver(var_type=var_type, var=var, pot_params=pot_params, ic=ic,
                           N_star=N_star, integrator=integrator, verbose=False)
        print(output)
        print("          N_tot: %.15f" % b.N_tot)
        print("         N_star: %.15f" % b.N_star)
        print()
        print("         a_init: %.15f" % b.ic[0])
        print("       phi_init: %.15f" % b.ic[1])
        print("      dphi_init: %.15e" % b.ic[2])
        print()
    return ic


def phi2efolds_planck(phi_0, var_type, var, pot_params, integrator='lsoda'):
    """
    Find total number of e-folds of inflation given phi_0
    :param phi_0: float
        phi_0 as in Handley et al. "Kinetic initial conditions for inflation"
    :param var_type: str
        Choice of independent variable, either 'time' or 'loga'.
    :param var: arraylike
        array of independent variable
    :param pot_params: list
        insert pot_params used by the potential
        where pot_params[0] is a string specifying the potential model
        'm2phi2' : V(phi) = m^2 phi^2
        'm2phi4' : V(phi) = m^2 phi^4
        'm2phi6' : V(phi) = m^2 phi^6
    :param integrator: str
        Choice of integrator, either of 'lsoda' (default), 'vode', 'zvode', 'dopri5', 'dop853'.
    :return: float
        N_tot = total number of e-folds of inflation
    """
    Pot = InflationaryPotential(pot_params=pot_params)
    dphi_0 = - np.sqrt(2 * MP**4 - 2 * Pot.V(phi_0))
    ic = np.array([1., phi_0, dphi_0])
    bkd = InflatonSolver(var_type=var_type, var=var, pot_params=pot_params, ic=ic,
                         N_star=0, integrator=integrator, verbose=False)
    if np.nanmax(np.abs(bkd.phi[-10:])) > 1.:
        print("phi might still be inflating, please increase time")
    return bkd.N_tot


def efolds2phi_planck(N_tot, phi_min, phi_max, var_type, var, pot_params, N_star=0,
                      integrator='lsoda', verbose=True):
    """
    Given the desired total number of e-folds of inflation N_tot,
    find the required initial conditions.
    :param N_tot: float
        total number of e-folds of inflation
    :param phi_min: float
        initial minimum guess for phi_0
    :param phi_max: float
        initial maximum guess for phi_0
    :param var_type: str
        Choice of independent variable, either 'time' or 'loga'.
    :param var: arraylike
        array of independent variable
    :param pot_params: list
        insert pot_params used by the potential
        where pot_params[0] is a string specifying the potential model
        'm2phi2' : V(phi) = m^2 phi^2
        'm2phi4' : V(phi) = m^2 phi^4
        'm2phi6' : V(phi) = m^2 phi^6
    :param float N_star: number of observable e-folds of inflation
        (since not important for determining the total number of e-folds
        it is fine to leave this at its default: 0)
    :param integrator: str
        Choice of integrator, either of 'lsoda' (default), 'vode', 'zvode', 'dopri5', 'dop853'.
    :param verbose: bool
        if True, prints out initial conditions
    :return: np.ndarray
        array of initial conditions: np.array([a_init, phi_init, dphi_init])
    """
    Pot = InflationaryPotential(pot_params=pot_params)

    def func(x): return phi2efolds_planck(phi_0=x, var_type=var_type, var=var,
                                          pot_params=pot_params, integrator=integrator) - N_tot

    new_phi_0, output = optimize.brentq(f=func, a=phi_min, b=phi_max, xtol=1e-15, full_output=True)

    dphi_0 = - np.sqrt(2 * MP**4 - 2 * Pot.V(new_phi_0))
    ic = np.array([1., new_phi_0, dphi_0])
    if verbose:
        bkd = InflatonSolver(var_type=var_type, var=var, pot_params=pot_params, ic=ic,
                             N_star=N_star, integrator=integrator, verbose=False)
        print(output)
        print("          N_tot: %.15f" % bkd.N_tot)
        print("         N_dagg: %.15f" % bkd.N_dagg)
        print("         N_star: %.15f" % bkd.N_star)
        print()
        print("         a_init: %.15f" % bkd.ic[0])
        print("       phi_init: %.15f" % bkd.ic[1])
        print("      dphi_init: %.15e" % bkd.ic[2])
        print()
    return ic
