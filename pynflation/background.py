from __future__ import print_function, division, absolute_import
from warnings import warn
import numpy as np
from scipy import integrate, interpolate
from pynflation.utils import nearest_idx
from pynflation.parameters import MP, K_STAR
from pynflation.potential import InflationaryPotential


class InflatonSolver(InflationaryPotential):
    """
    Class to calculate the background variables during inflation:
    Also sets up the value aH_star necessary to calibrate the primordial power spectrum.

    Parameters
    ----------
        var_type : choice of independent variable, either 'time' or 'loga'
        var : array of independent variable
        pot_params : parameters used by the potential function
        ic : array of initial conditions [a0, phi0, dphi0]
        N_star : desired number of e-folds of inflation after horizon crossing
        integrator : scipy's ODE integrator
        verbose : optional verbose printing

    Attributes
    ----------
        a = scale factor
        phi = inflaton field
        dphidt = first derivative of inflaton field w.r.t. time t
        dphidloga = first derivative of inflaton field w.r.t. log(a)
        H = Hubble parameter
        idx_infl = indices of inflation, i.e. indices i where w[i] < -1/3
        N_tot = total number of e-folds of inflation
        N_dagg = e-folds before horizon crossing, i.e. N_dagg = N_tot - N_star
        aH_star = a*H at pivot scale K_STAR where N_star e-folds are left
        aH = product a*H of scale factor and Hubble parameter
        N = number of e-folds of inflation defined as N=log(a)
        rho_phi = energy density associated with the inflaton field phi
        p_phi = pressure associated with the inflaton field phi
        w_phi = equation of state parameter associated with the inflaton field phi
        K_phi = fraction of kinetic energy to total energy density rho

    Methods
    -------
        aH2k : conversion function from a*H to wavenumber k
        k2aH : conversion function from wavenumber k to a*H
        calc_pk_approx : calculate approximate PPS as (H^2/2pi dphi)^2

    """
    def __init__(self, var_type, var, pot_params, ic, N_star,
                 integrator='lsoda', verbose=True, do_interval=True):
        """
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
        :param ic: np.ndarray(3)
            array of initial conditions: np.array([a_init, phi_init, dphi_init])
            Note that dphi_init corresponds to dphi/dt or dphi/dloga depending on var_type.
        :param N_star: float
            Desired number of (observable) e-folds of inflation,
            i.e. number of e-folds after horizon crossing for pivot scale K_STAR.
        :param integrator: str
            Choice of integrator, either of 'lsoda' (default), 'vode', 'zvode', 'dopri5', 'dop853'.
        :param verbose: bool
            if True, prints out initial conditions
        """
        InflationaryPotential.__init__(self, pot_params)
        verboseprint = print if verbose else lambda *a, **k: None
        self.verboseprint = verboseprint
        self.var_type = var_type
        self.var = var  # type: np.ndarray
        self.ic = ic
        self.N_star = N_star  # number of e-folds after horizon crossing
        self.integrator = integrator
        self.do_interval = do_interval

        self._func_setup()
        self._background_evolution()

        if do_interval:
            verboseprint("Initial conditions used: ")
            verboseprint("a_init = %.3f  phi_init = %.3f  dphi_init = %.3e" % tuple(self.ic))
            verboseprint("N_tot  = ", self.N_tot)
            verboseprint("N_dagg = ", self.N_tot - self.N_star)
            verboseprint("N_star = ", self.N_star)
            verboseprint()

    def _func_setup(self):
        """
        Set up which ODE function and jacobian to use later on in scipy's ODE integrator,
        i.e. w.r.t. time t or log(a).
        """
        assert len(self.ic) == 3, "There were %s initial conditions provided, " \
                                  "but it should have been 3." % len(self.ic)
        if self.var_type == 'time':
            self.func = background_func_t
            self.Dfun = background_jacobian_t
        elif self.var_type == 'loga':
            self.func = background_func_a
            self.Dfun = background_jacobian_a
        else:
            raise Exception("var_type is %s, but only 'time' and 'loga' supported."
                            % self.var_type)

    def _background_evolution(self):
        """
        This is where the actual integration (call to scipy's ODE integrator) of the
        background variables happens.
        We have a system of 1st order ODEs corresponding to the initial conditions vector,
        such that ys[i] = [a[i], phi[i], dphi[i]].
        """
        ys = np.empty((len(self.var), len(self.ic)))  # type: np.ndarray
        ys.fill(np.nan)
        ys[0] = self.ic

        integrator_params = [self.pot_params]
        ig = integrate.ode(f=self.func, jac=self.Dfun)
        ig.set_integrator(self.integrator)  # , atol=1e-12, rtol=1e-10, nsteps=1000)
        ig.set_initial_value(y=self.ic, t=self.var[0])
        ig.set_f_params(*integrator_params)
        ig.set_jac_params(*integrator_params)
        i = 0
        while ig.successful() and ig.t <= self.var[-1] and i + 1 < len(self.var) and \
                ig.t == self.var[i]:
            # if i > 10 and np.all([np.abs(ys[i-10:i, 1]) < 0.01]):
            #     break
            ys[i + 1, :] = ig.integrate(t=self.var[i+1])
            i += 1

        self._background_setup(ys)
        if self.do_interval:
            self._inflation_interval()

    def _background_setup(self, ys):
        """
        Using the result ys from the integration we store the background variables:

        a = scale factor
        phi = inflaton field
        dphidt = 1st derivative of phi w.r.t. time t
        dphidloga = 1st derivative of phi w.r.t. log(a)
        H = Hubble parameter
        :param ys: np.ndarray(num_var, 3)
            result of ODE integration such that ys[i] = [a[i], phi[i], dphi[i]]
        """
        self.a = ys[:, 0]  # type: np.ndarray
        self.phi = ys[:, 1]  # type: np.ndarray

        if self.var_type == 'time':
            self.dphidt = ys[:, 2]  # type: np.ndarray
            self.H = np.sqrt((self.dphidt**2 / (2. * MP**2) + self.V(self.phi))
                             / (3. * MP**2))  # type: np.ndarray
            self.dphidloga = self.dphidt / self.H  # type: np.ndarray
        elif self.var_type == 'loga':
            self.dphidloga = ys[:, 2]  # type: np.ndarray
            self.H = np.sqrt(self.V(self.phi) / (3. * MP**2) /
                             (1. - self.dphidloga**2 / (6. * MP**2)))  # Hubble parameter
            self.dphidt = self.H * self.dphidloga  # type: np.ndarray
        else:
            raise Exception("var_type is %s, but only 'time' and 'loga' supported."
                            % self.var_type)

    def _inflation_interval(self):
        """
        Here we determine the onset and end of inflation
        and derive the corresponding number of e-folds.

        idx_infl = indices of inflation, i.e. indices i where w[i] < -1/3
        N_tot = total number of e-folds of inflation
        N_dagg = number of e-folds before horizon crossing, i.e. N_dagg = N_tot - N_star
        aH_star = value of a*H at pivot scale K_STAR where N_star e-folds of inflation are left
        """
        assert np.nanmin(self.K_phi) >= 0, "K should be strictly >= zero, but K[%d] = %.2e" % \
                                           (np.argmin(self.K_phi < 0).ravel()[0],
                                            self.K_phi[np.argmin(self.K_phi < 0).ravel()[0]])
        inflation_thresh = np.argwhere(np.diff(np.sign(self.K_phi - 1. / 3.))).ravel()
        if self.K_phi[0] <= 1. / 3.:  # directly starting in inflation
            if self.K_phi[0] <= 0.01:
                print("Starting during slow roll inflation. (SR)")
            else:
                self.verboseprint("Starting during fast roll inflation. In this region neither "
                                  "kinetic dominance nor slow roll initial conditions apply. "
                                  "Please adjust initial conditions or starting time.")
                if self.N_star is not 0:
                    raise Exception("Neither in slow roll nor in kinetic dominance")
            idx_beg = 0  # directly starting in inflation
            loga_beg = np.log(self.a[0])  # directly starting in inflation
            if np.nanmax(self.K_phi) < 1. / 3.:  # inflation not ending
                print("Inflation does not end. Try extending time?")
                idx_end = len(self.K_phi) - 1
                loga_end = np.log(self.a[-1])
            else:  # inflation ended at some point
                idx_end = inflation_thresh[0] + 1
                K2loga_end = interpolate.interp1d(
                    self.K_phi[inflation_thresh[0] - 1:inflation_thresh[0] + 3],
                    np.log(self.a[inflation_thresh[0]-1:inflation_thresh[0] + 3]),
                    kind='cubic'
                )
                loga_end = K2loga_end([1. / 3.])[0]
        elif self.K_phi[0] <= 1.:  # starting during kinetic dominance (KD)
            # print("Starting during kinetic dominance. (KD)")
            assert len(inflation_thresh) >= 1, "Inflation does not start. Extend time?"
            assert self.K_phi[inflation_thresh[0] + 1] <= 1. / 3., \
                "K did not drop below 1/3, i.e. inflation did not start. Extend time?"
            idx_beg = inflation_thresh[0] + 1
            K2loga_beg = interpolate.interp1d(
                self.K_phi[inflation_thresh[0] - 1:inflation_thresh[0] + 3],
                np.log(self.a[inflation_thresh[0]-1:inflation_thresh[0] + 3]),
                kind='cubic'
            )
            loga_beg = K2loga_beg([1. / 3.])[0]
            if len(inflation_thresh) < 2:  # inflation not ending
                idx_end = len(self.K_phi) - 1
                loga_end = np.log(self.a[-1])
            else:  # inflation ended at some point
                idx_end = inflation_thresh[1] + 1
                K2loga_end = interpolate.interp1d(
                    self.K_phi[inflation_thresh[1] - 1:inflation_thresh[1] + 3],
                    np.log(self.a[inflation_thresh[1]-1:inflation_thresh[1] + 3]),
                    kind='cubic'
                )
                loga_end = K2loga_end([1. / 3.])[0]
        else:
            raise Exception("Unexpected: K = %s > 1. Should be <= 1!" % self.K_phi[0])

        self.idx_infl = list(range(idx_beg, idx_end + 1))  # index list of inflation range
        self.N_tot = loga_end - loga_beg  # total number of e-folds of inflation
        self.N_dagg = self.N_tot - self.N_star  # number of e-folds before horizon crossing
        loga2aH = interpolate.interp1d(np.log(self.a), self.aH)
        self.aH_star = loga2aH(loga_end - self.N_star)  # for calibration with pivot scale K_STAR

    def aH2k(self, aH):
        """
        Calibration function moving from a*H to wavenumber k [Mpc-1],
        where a is the scale factor and H is the Hubble parameter.
        :param aH: float or arraylike
            product a*H in [Mpc-1]
        :return: wavenumber k
        """
        k = K_STAR * aH / self.aH_star
        return k

    def k2aH(self, k):
        """
        Calibration function moving from wavenumber k to a*H,
        where a is the scale factor and H is the Hubble parameter.
        :param k: float or arraylike
            wavenumber in [Mpc-1]
        :return: product a*H
        """
        aH = self.aH_star * k / K_STAR
        return aH

    @property
    def aH(self):
        """
        :return: arraylike
            product a*H of scale factor and Hubble parameter
        """
        return self.a * self.H  # type: np.ndarray

    @property
    def N(self):
        """
        :return: arraylike
            Number of e-folds of inflation defined as N=log(a)
        """
        return np.log(self.a)  # type: np.ndarray

    @property
    def rho_phi(self):
        """
        Energy density associated with the inflaton field phi.
        rho = dphidt**2 / 2 + V(phi)
        :return: arraylike
        """
        return self.dphidt**2 / 2. + self.V(self.phi)  # type: np.ndarray

    @property
    def p_phi(self):
        """
        Pressure associated with the inflaton field phi.
        p = dphidt**2 / 2 - V(phi)
        :return: arraylike
        """
        return self.dphidt**2 / 2. - self.V(self.phi)  # type: np.ndarray

    @property
    def w_phi(self):
        """
        Equation of state parameter associated with the inflaton field phi.
        w = p / rho,
        where p is the pressure and rho is the energy density of the inflaton field.
        :return: arraylike
        """
        return self.p_phi / self.rho_phi  # type: np.ndarray

    @property
    def K_phi(self):
        """
        Fraction of kinetic energy to total energy density rho.
        K = dphidt**2/2 / rho
        :return: arraylike
        """
        return self.dphidt**2 / 2. / self.rho_phi  # type: np.ndarray

    def calc_pk_approx(self):
        """
        Calculate the approximate power spectrum according to
        P(k) = (H^2 / 2pi dphidt)^2
        :return: np.ndarray, np.ndarray
            k, pk: the wavenumber and the approximate power spectrum
        """
        warn("calc_pk_approx() is deprecated, use calc_analytic_scalar_power() instead.")
        k = self.aH2k(self.aH)
        pk = (self.H**2 / (2. * np.pi * self.dphidt))**2
        return k[self.idx_infl], pk[self.idx_infl]

    def calc_analytic_scalar_power(self):
        """
        Calculate the approximate scalar power spectrum according to
        P(k) = (H^2 / 2pi dphidt)^2
        :return: np.ndarray, np.ndarray
            k, scalar_power: the wavenumber and the approximate power spectrum
        """
        k = self.aH2k(self.aH)
        scalar_power = (self.H**2 / (2. * np.pi * self.dphidt))**2
        return k[self.idx_infl], scalar_power[self.idx_infl]

    def calc_analytic_tensor_power(self):
        """
        Calculate the approximate tensor power spectrum according to
        P(k) = 2 (H / pi m_p)^2
        :return: np.ndarray, np.ndarray
            k, tensor_power: the wavenumber and the approximate power spectrum
        """
        k = self.aH2k(self.aH)
        tensor_power = 2. * (self.H / (np.pi * MP))**2
        return k[self.idx_infl], tensor_power[self.idx_infl]

    def calc_pk_approx_slope(self, k1, k2):
        k, pk = self.calc_analytic_scalar_power()
        k2pk = interpolate.interp1d(k, pk)
        # idx1 = nearest_idx(k, k1)
        # idx2 = nearest_idx(k, k2)
        # p1 = pk[idx1]
        # p2 = pk[idx2]
        # k1 = k[idx1]
        # k2 = k[idx2]
        p1 = k2pk(k1)
        p2 = k2pk(k2)
        return self.calc_loglog_slope(p1, p2, k1, k2)

    def calc_approx_ns(self):
        """
        Calculate the spectral index n_s from the analytic approximate scalar power spectrum P(k).
        :return: float
            ns: the spectral index
        """
        k, scalar_pk = self.calc_analytic_scalar_power()
        idx = nearest_idx(k, K_STAR)
        ns = 1 + self.calc_loglog_slope(scalar_pk[idx-1], scalar_pk[idx+1], k[idx-1], k[idx+1])
        return ns

    def calc_approx_r(self):
        """
        Calculate the tensor to scalar ratio r from the analytic approximate tensor and scalar
        power spectra.
        :return: float
            r: tensor to scalar ratio
        """
        k, scalar_power = self.calc_analytic_scalar_power()
        k, tensor_power = self.calc_analytic_tensor_power()
        idx = nearest_idx(k, K_STAR)
        r = tensor_power[idx] / scalar_power[idx]
        return r

    @staticmethod
    def calc_loglog_slope(y1, y2, x1, x2):
        return (np.log(y2) - np.log(y1)) / (np.log(x2) - np.log(x1))


def background_func_t(t0, y, pot_params):
    """
    Background evolution equations of inflation
    written as 1st order ODE in a, N, phi, dphidt,
    where t is physical (not conformal) time.

    :param t0: float
        starting time in terms of physical time t
    :param y: arraylike
        y[0] : a
        y[1] : phi
        y[2] : dphidt
    :param pot_params: list
        insert pot_params used by the potential
        where pot_params[0] is a string specifying the potential model
        'm2phi2' : V(phi) = m^2 phi^2
        'm2phi4' : V(phi) = m^2 phi^4
        'm2phi6' : V(phi) = m^2 phi^6
    :return: arraylike (same as y)
        derivative dy/dt for parameter y
    """
    del t0
    Pot = InflationaryPotential(pot_params=pot_params)

    H = np.sqrt((y[2]**2 / 2. + Pot.V(y[1])) / (3. * MP**2))

    dydt = np.zeros(3)
    dydt[0] = H * y[0]
    dydt[1] = y[2]
    dydt[2] = - 3. * y[2] * H - Pot.dVdphi(y[1])

    return dydt


def background_jacobian_t(t0, y, pot_params):
    """
    Jacobian to corresponding background function background_func_t.

    :param t0: float
        starting time in terms of physical time t
    :param y: arraylike
        y[0] : a
        y[1] : phi
        y[2] : dphidt
    :param pot_params: list
        insert pot_params used by the potential
        where pot_params[0] is a string specifying the potential model
        'm2phi2' : V(phi) = m^2 phi^2
        'm2phi4' : V(phi) = m^2 phi^4
        'm2phi6' : V(phi) = m^2 phi^6
    :return: arraylike (same as y)
        jacobian, i.e. Df(y) where f is the corresponding background function of parameter y
    """
    del t0
    Pot = InflationaryPotential(pot_params=pot_params)

    H = np.sqrt((y[2] ** 2 / 2. + Pot.V(y[1])) / (3. * MP ** 2))

    J = np.zeros((3, 3))

    J[0][0] = H
    J[0][1] = y[0] * Pot.dVdphi(y[1]) / (6. * H * MP**2)
    J[0][2] = y[0] * y[2] / (6. * H * MP**2)

    J[1][2] = 1.

    J[2][1] = - 3. * y[2] * Pot.dVdphi(y[1]) / (6. * H * MP**2) - Pot.d2Vdphi2(y[1])
    J[2][2] = - 3. * H - 3. * y[2]**2 / (6. * H * MP**2)

    return J

