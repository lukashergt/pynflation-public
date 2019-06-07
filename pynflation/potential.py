from __future__ import print_function, division, absolute_import
from warnings import warn
import numpy as np
from pynflation.parameters import MP


class InflationaryPotential:
    """
    Class to set up methods all about the inflationary potential V(phi),
    i.e. potential itself, derivatives...

    Parameters
    ----------
        pot_params : list
            Contains parameters used by the potential function,
            where pot_params[0] is a string specifying one of the following potential models:
            'm2phi2' : V(phi) = m^2 phi^2
            'm2phi4' : V(phi) = m^2 phi^4
            'm2phi6' : V(phi) = m^2 phi^6
            'phi2/3' : V(phi) = m^2 phi^(2/3)
            'phi4/3' : V(phi) = m^2 phi^(4/3)
            'starobinsky' : V(phi) = lam^4 * (1 - e^(-sqrt(2/3)phi))^2
            'phi2phi4' : V(phi) = lam * (phi^2 - phi_min^2)^2
            'hilltop' : V(phi) = lam^4 * (1 - phi^2/mu^2)^2
            The inflaton mass m is measured in Planck masses MP.

    Methods
    -------
        V : inflationary potential function V(phi)
        inv_V : inverse potential function, i.e. phi(V)
        dVdphi : 1st derivative of V(phi)
        dVV : dVdphi / V
        d2Vdphi2 : 2nd derivative of V(phi)
        d3Vdphi3 : 3rd derivative of V(phi)

    """
    _power_law_pot = ['m2phi2', 'm2phi4', 'm2phi6', 'phi2/3', 'phi4/3']
    _possible_pot_models = _power_law_pot + ['starobinsky', 'nat_10', 'phi2phi4', 'hilltop']

    def __init__(self, pot_params):
        """
        :param pot_params: list
            Contains parameters used by the potential function.
            For details refer to the class descriptions.
        """
        assert pot_params[0] in self._possible_pot_models, ("%s not a possible potential model."
                                                            % pot_params[0])
        self.pot_params = pot_params
        self.pot_model = pot_params[0]

        if self.pot_model in self._power_law_pot:
            assert len(self.pot_params) == 2, ("power law potentials take exactly 1 parameter m "
                                               "(mass), %s parameters given."
                                               % (len(self.pot_params) - 1))
            self.m = self.pot_params[1] * MP
            if self.pot_model == 'm2phi2':
                self.n = 2
            elif self.pot_model == 'm2phi4':
                self.n = 4
            elif self.pot_model == 'm2phi6':
                self.n = 6
            elif self.pot_model == 'phi2/3':
                self.n = 2. / 3.
            elif self.pot_model == 'phi4/3':
                self.n = 4. / 3.
            else:
                raise NotImplementedError("pot_model=%s not an implemented power law potential."
                                          % self.pot_model)
        elif self.pot_model == 'starobinsky':
            assert len(self.pot_params) == 2, ("'starobinsky' potential takes 1 parameter lam2"
                                               "%s parameters given." % (len(self.pot_params) - 1))
            self.lam2 = self.pot_params[1] * MP**2
        elif self.pot_model == 'nat_10':
            assert len(self.pot_params) == 2, ("nat_10 potential takes exactly 1 parameter lam "
                                               "(mass), %s parameters given."
                                               % (len(self.pot_params) - 1))
            self.lam = self.pot_params[1]
            self.f = 10. * MP
        elif self.pot_model == 'phi2phi4':
            assert len(self.pot_params) == 3, ("phi2phi4 potential takes exactly 2 parameters "
                                               "lam (lambda) and "
                                               "phi_min (phi at potential minimum), "
                                               "%s parameters given." % (len(self.pot_params) - 1))
            self.lam = self.pot_params[1]
            self.phi_min = self.pot_params[2] * MP
        elif self.pot_model == 'hilltop':
            assert len(self.pot_params) == 3, ("hilltop potential takes exactly 2 parameters "
                                               "lam (lambda) and "
                                               "phi_min (phi at potential minimum), "
                                               "%s parameters given." % (len(self.pot_params) - 1))
            self.lam2 = self.pot_params[1] * MP**2
            self.mu = self.pot_params[2] * MP
        else:
            raise NotImplementedError("pot_model=%s not implemented." % self.pot_model)

    def V(self, phi):
        """
        Inflationary potential V(phi) given phi
        as specified by the choice of the potential type self.pot_model.

        :param Union[float, np.ndarray] phi: inflaton field phi
        :return: arraylike (same as phi)
            potential V(phi) at phi
        """
        if self.pot_model in self._power_law_pot:
            return self.m**2 * np.abs(phi)**self.n
        elif self.pot_model == 'starobinsky':
            return self.lam2**2 * (1 - np.exp(- phi * np.sqrt(2. / 3.) / MP))**2
        elif self.pot_model == 'nat_10':
            return self.lam**4 * (1. + np.cos(phi / self.f))
        elif self.pot_model == 'phi2phi4':
            return self.lam * (phi**2 - self.phi_min**2)**2
        elif self.pot_model == 'hilltop':
            return self.lam2**2 * (1 - phi**2/self.mu**2)**2
        else:
            raise Exception("Potential %s not listed for method V." % self.pot_model)

    def inv_V(self, Vphi):
        """
        Inverted inflationary potential phi(V) given the value for V(phi),
        as specified by the choice of the potential type self.pot_model.

        :param Vphi: arraylike
            potential value V
        :return: arraylike (same as Vphi)
            phi value at given potential value V(phi)
        """
        if np.any(np.atleast_1d(Vphi) < 0):
            warn("Negative values for power law potential unexpected.")
        if self.pot_model in self._power_law_pot:
            phi1 = (Vphi / self.m**2)**(1. / self.n)
            return -phi1, phi1
        elif self.pot_model == 'phi2phi4':
            return ((Vphi / self.lam)**(1. / 2.) + self.phi_min**2)**(1. / 2.)
        elif self.pot_model == 'starobinsky':
            phi1 = - np.sqrt(3. / 2.) * MP * np.log(1. + np.sqrt(Vphi) / self.lam2)
            phi2 = - np.sqrt(3. / 2.) * MP * np.log(1. - np.sqrt(Vphi) / self.lam2)
            return phi1, phi2
        elif self.pot_model == 'hilltop':
            phi1 = self.mu * np.sqrt(1. - np.sqrt(Vphi) / self.lam2)
            phi2 = self.mu * np.sqrt(1. + np.sqrt(Vphi) / self.lam2)
            return -phi2, -phi1, phi1, phi2
        else:
            raise NotImplementedError("pot_model=%s not listed for method inv_V." % self.pot_model)

    def dVdphi(self, phi):
        """
        First derivative dV/dphi given phi
        of the inflationary potential V(phi)
        as specified by the choice of the potential type self.pot_model.

        :param phi: arraylike
            inflaton field phi
        :return: arraylike (same as phi)
            derivative dV/dphi at phi
        """
        if self.pot_model in self._power_law_pot:
            if self.pot_model in ['phi2/3', 'phi4/3']:
                # FIXME: actually not sure how to handle these two potentials here
                return np.sign(phi) * self.n * self.m**2 * np.abs(phi)**(self.n - 1)
            else:
                return self.n * self.m**2 * phi**(self.n - 1)
        elif self.pot_model == 'starobinsky':
            return 2. / MP * np.sqrt(2. / 3.) * np.exp(- phi * np.sqrt(2. / 3.) / MP) \
                   * self.lam2**2 * (1. - np.exp(- phi * np.sqrt(2. / 3.) / MP))
        elif self.pot_model == 'nat_10':
            return - self.lam**4 * np.sin(phi / self.f) / self.f
        elif self.pot_model == 'phi2phi4':
            return 4. * self.lam * phi * (phi**2 - self.phi_min**2)
        elif self.pot_model == 'hilltop':
            return -4 * self.lam2**2 * phi / self.mu**2 * (1 - phi**2 / self.mu**2)
        else:
            raise NotImplementedError("pot_model=%s not listed for method dVdphi."
                                      % self.pot_model)

    def dVV(self, phi):
        """
        dV/dphi / V

        :param phi: arraylike
            inflaton field phi
        :return: arraylike (same as phi)
            dV/dphi / V at phi
        """
        if self.pot_model in self._power_law_pot:
            return self.n / phi
        elif self.pot_model == 'starobinsky':
            return 2. / MP * np.sqrt(2. / 3.) / (np.exp(phi * np.sqrt(2. / 3.) / MP) - 1.)
        elif self.pot_model == 'phi2phi4':
            return phi / (phi**2 - self.phi_min**2)
        else:
            raise NotImplementedError("pot_model=%s not listed for method dVV." % self.pot_model)

    def d2Vdphi2(self, phi):
        """
        Second derivative d2V/dphi2 given phi
        of the inflationary potential V(phi)
        as specified by the choice of the potential type self.pot_model.

        :param phi: arraylike
            inflaton field phi
        :return: arraylike (same as phi)
            second derivative d2V/dphi2 at phi
        """
        if self.pot_model in self._power_law_pot:
            return self.n * (self.n - 1) * self.m**2 * phi**(self.n - 2)
        elif self.pot_model == 'starobinsky':
            return - 4. / (3. * MP**2) * np.exp(- phi * np.sqrt(2. / 3.) / MP) \
                   * self.lam2**2 * (1. - 2. * np.exp(- phi * np.sqrt(2. / 3.) / MP))
        elif self.pot_model == 'nat_10':
            return - self.lam**4 * np.cos(phi / self.f) / self.f**2
        elif self.pot_model == 'phi2phi4':
            return self.lam * (3. * phi**2 - self.phi_min**2)
        elif self.pot_model == 'hilltop':
            return -4 * self.lam2**2 / self.mu**2 * (1 - 3 * phi**2 / self.mu**2)
        else:
            raise NotImplementedError("pot_model=%s not listed for method d2Vdphi2."
                                      % self.pot_model)

    def d3Vdphi3(self, phi):
        """
        Third derivative d3V/dphi3 given phi
        of the inflationary potential V(phi)
        as specified by the choice of the potential type self.pot_model.

        :param phi: arraylike
            inflaton field phi
        :return: arraylike (same as phi)
            third derivative d3V/dphi3 at phi
        """
        if self.pot_model == 'm2phi2':
            return 0.
        elif self.pot_model in self._power_law_pot:
            return self.n * (self.n - 1) * (self.n - 2) * self.m**2 * phi**(self.n - 3)
        elif self.pot_model == 'starobinsky':
            return 4. / (3. * MP**3) * np.sqrt(2. / 3.) * np.exp(- phi * np.sqrt(2. / 3.) / MP) \
                   * self.lam2**2 * (1. - 4. * np.exp(- phi * np.sqrt(2. / 3.) / MP))
        elif self.pot_model == 'nat_10':
            return self.lam**4 * np.cos(phi / self.f) / self.f**3
        elif self.pot_model == 'phi2phi4':
            return 6. * self.lam * phi
        else:
            raise NotImplementedError("pot_model=%s not listed for method d3Vdphi3."
                                      % self.pot_model)
