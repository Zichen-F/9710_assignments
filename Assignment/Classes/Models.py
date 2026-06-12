import numpy as np

class DiffusionModel:
    """Class defining a diffusion model"""

    def __init__(self, grid, phi, gamma, west_bc, east_bc):
        """Constructor"""
        self._grid = grid
        self._phi = phi
        self._gamma = gamma
        self._west_bc = west_bc
        self._east_bc = east_bc

    def add(self, coeffs):
        """Function to add diffusion terms to coefficient arrays"""

        # Calculate the west and east face diffusion flux terms for each face
        flux_w = - self._gamma*self._grid.Aw*(self._phi[1:-1]-self._phi[0:-2])/self._grid.dx_WP
        flux_e = - self._gamma*self._grid.Ae*(self._phi[2:]-self._phi[1:-1])/self._grid.dx_PE

        # Calculate the linearization coefficients
        coeffW = - self._gamma*self._grid.Aw/self._grid.dx_WP
        coeffE = - self._gamma*self._grid.Ae/self._grid.dx_PE
        coeffP = - coeffW - coeffE

        # Modify the linearization coefficients on the boundaries
        coeffP[0] += coeffW[0]*self._west_bc.coeff()
        coeffP[-1] += coeffE[-1]*self._east_bc.coeff()

        # Zero the boundary coefficients that are not used
        coeffW[0] = 0.0
        coeffE[-1] = 0.0

        # Calculate the net flux from each cell
        flux = flux_e - flux_w

        # Add to coefficient arrays
        coeffs.accumulate_aP(coeffP)
        coeffs.accumulate_aW(coeffW)
        coeffs.accumulate_aE(coeffE)
        coeffs.accumulate_rP(flux)

        # Return the modified coefficient array
        return coeffs

class SurfaceConvectionModel:
    """Class defining a surface convection model"""

    def __init__(self, grid, T, ho, To):
        """Constructor"""
        self._grid = grid
        self._T = T
        self._ho = ho
        self._To = To

    def add(self, coeffs):
        """Function to add surface convection terms to coefficient arrays"""

        # Calculate the source term
        source = self._ho*self._grid.Ao*(self._T[1:-1] - self._To)

        # Calculate the linearization coefficients
        coeffP = self._ho*self._grid.Ao

        # Add to coefficient arrays
        coeffs.accumulate_aP(coeffP)
        coeffs.accumulate_rP(source)

        return coeffs
        
class FirstOrderTransientModel:
    """Class defining a first order implicit transient model"""

    def __init__(self, grid, T, Told, rho, cp, dt):
        """Constructor"""
        self._grid = grid
        self._T = T
        self._Told = Told
        self._rho = rho
        self._cp = cp
        self._dt = dt

    def add(self, coeffs):
        """Function to add transient term to coefficient arrays"""

        # Calculate the transient term
        rp_coeff = (self._rho * self._cp * self._grid.vol/ self._dt) * (self._T[1:-1] - self._Told[1:-1])

        # Calculate the linearization coefficient
        ap_coeff = (self._rho * self._cp * self._grid.vol/ self._dt)
        
        # Add to coefficient arrays
        coeffs.accumulate_aP(ap_coeff)
        coeffs.accumulate_rP(rp_coeff)
        
        return coeffs

class SecondOrderTransientModel:
    """Class defining a second order implicit transient model"""

    def __init__(self, grid, T, Told, Told2, rho, cp, dt):
        """Constructor"""
        self._grid = grid
        self._T = T
        self._Told = Told
        self._Told2 = Told2
        self._rho = rho
        self._cp = cp
        self._dt = dt

    def add(self, coeffs):
        """Function to add transient term to coefficient arrays"""
        
        # Calculate the transient term rhp * cp * V *(3T - 4Told + T^Told2) / (2 dt)
        rp_coeff = self._rho * self._cp * self._grid.vol * (3.0 * self._T[1:-1] - 4.0 * self._Told[1:-1] + self._Told2[1:-1]) / (2.0 * self._dt)

        # Calculate the linearization coefficient 3 rho cp V / (2 dt)
        ap_coeff = 3.0 * self._rho * self._cp * self._grid.vol / (2.0 * self._dt)
        
        # Add to coefficient arrays
        coeffs.accumulate_aP(ap_coeff)
        coeffs.accumulate_rP(rp_coeff)
        
        return coeffs
class UpwindAdvectionModel:
    """Class defining an upwind advection model"""

    def __init__(self, grid, phi, Uhe, rho, const, west_bc, east_bc):
        """Constructor"""
        self._grid = grid
        self._phi = phi
        self._Uhe = Uhe
        self._rho = rho
        self._const = const
        self._west_bc = west_bc
        self._east_bc = east_bc
        self._alphae = np.zeros(self._grid.ncv+1)
        self._phie = np.zeros(self._grid.ncv+1)

    def add(self, coeffs):
        """Function to add diffusion terms to coefficient arrays"""

        # Calculate the weighting factors
        for i in range(self._grid.ncv+1):
            if self._Uhe[i] >= 0:
                self._alphae[i] = 1
            else:
                self._alphae[i] = -1
        
        # Calculate the east integration point values (including both boundaries)
        self._phie = (1 + self._alphae)/2*self._phi[0:-1] + (1 - self._alphae)/2*self._phi[1:]
        
        # Calculate the face mass fluxes
        mdote = self._rho*self._Uhe*self._grid.Af
        
        # Calculate the west and east face advection flux terms for each face
        flux_w = self._const*mdote[:-1]*self._phie[:-1]
        flux_e = self._const*mdote[1:]*self._phie[1:]
        
        # Calculate mass imbalance term
        imbalance = - self._const*mdote[1:]*self._phi[1:-1] + self._const*mdote[:-1]*self._phi[1:-1]
          
        # Calculate the linearization coefficients
        coeffW = - self._const*mdote[:-1]*(1 + self._alphae[:-1])/2
        coeffE = self._const*mdote[1:]*(1 - self._alphae[1:])/2
        coeffP = - coeffW - coeffE

        # Modify the linearization coefficients on the boundaries
        coeffP[0] += coeffW[0]*self._west_bc.coeff()
        coeffP[-1] += coeffE[-1]*self._east_bc.coeff()

        # Zero the boundary coefficients that are not used
        coeffW[0] = 0.0
        coeffE[-1] = 0.0

        # Calculate the net flux from each cell
        flux = flux_e - flux_w

        # Add to coefficient arrays
        coeffs.accumulate_aP(coeffP)
        coeffs.accumulate_aW(coeffW)
        coeffs.accumulate_aE(coeffE)
        coeffs.accumulate_rP(flux)
        coeffs.accumulate_rP(imbalance)

        # Return the modified coefficient array
        return coeffs
        
class CDSAdvectionModel:
    """Class defining a deferred-correction CDS advection model"""

    def __init__(self, grid, phi, Uhe, rho, const, west_bc, east_bc):
        """Constructor"""
        self._grid = grid
        self._phi = phi
        self._Uhe = Uhe
        self._rho = rho
        self._const = const
        self._west_bc = west_bc
        self._east_bc = east_bc

        self._alphae = np.zeros(self._grid.ncv + 1)
        self._phie_uds = np.zeros(self._grid.ncv + 1)
        self._phie_cds = np.zeros(self._grid.ncv + 1)

    def add(self, coeffs):
        """Function to add advection terms to coefficient arrays"""

        # Calculate the weighting factors
        for i in range(self._grid.ncv + 1):
            if self._Uhe[i] >= 0:
                self._alphae[i] = 1
            else:
                self._alphae[i] = -1

        # Calculate the UDS face values
        self._phie_uds = (
            (1 + self._alphae) / 2 * self._phi[0:-1]
            + (1 - self._alphae) / 2 * self._phi[1:]
        )

        # Calculate the CDS face values
        self._phie_cds = 0.5 * (self._phi[0:-1] + self._phi[1:])

        # Calculate the face mass fluxes
        mdote = self._rho * self._Uhe * self._grid.Af

        # Calculate UDS fluxes
        flux_w_uds = self._const * mdote[:-1] * self._phie_uds[:-1]
        flux_e_uds = self._const * mdote[1:]  * self._phie_uds[1:]

        flux_uds = flux_e_uds - flux_w_uds

        # Calculate CDS fluxes
        flux_w_cds = self._const * mdote[:-1] * self._phie_cds[:-1]
        flux_e_cds = self._const * mdote[1:]  * self._phie_cds[1:]

        flux_cds = flux_e_cds - flux_w_cds

        # Calculate mass imbalance term
        imbalance = (
            - self._const * mdote[1:]  * self._phi[1:-1]
            + self._const * mdote[:-1] * self._phi[1:-1]
        )

        # Calculate the linearization coefficients using UDS
        coeffW = - self._const * mdote[:-1] * (1 + self._alphae[:-1]) / 2
        coeffE =   self._const * mdote[1:]  * (1 - self._alphae[1:])  / 2
        coeffP = - coeffW - coeffE

        # Modify the linearization coefficients on the boundaries
        coeffP[0] += coeffW[0] * self._west_bc.coeff()
        coeffP[-1] += coeffE[-1] * self._east_bc.coeff()

        # Zero the boundary coefficients that are not used
        coeffW[0] = 0.0
        coeffE[-1] = 0.0

        # Deferred correction
        flux = flux_uds + (flux_cds - flux_uds)

        # Add to coefficient arrays
        coeffs.accumulate_aP(coeffP)
        coeffs.accumulate_aW(coeffW)
        coeffs.accumulate_aE(coeffE)
        coeffs.accumulate_rP(flux)
        coeffs.accumulate_rP(imbalance)

        # Return the modified coefficient array
        return coeffs
        
class QUICKAdvectionModel:
    """Class defining a deferred-correction QUICK advection model"""

    def __init__(self, grid, phi, Uhe, rho, const, west_bc, east_bc):
        """Constructor"""
        self._grid = grid
        self._phi = phi
        self._Uhe = Uhe
        self._rho = rho
        self._const = const
        self._west_bc = west_bc
        self._east_bc = east_bc

        self._alphae = np.zeros(self._grid.ncv + 1)
        self._phie_uds = np.zeros(self._grid.ncv + 1)
        self._phie_quick = np.zeros(self._grid.ncv + 1)

    def add(self, coeffs):
        """Function to add deferred-correction QUICK advection terms"""

        # Calculate UDS weighting factors from flow direction
        for i in range(self._grid.ncv + 1):
            if self._Uhe[i] >= 0.0:
                self._alphae[i] = 1.0
            else:
                self._alphae[i] = -1.0

        # UDS face values
        self._phie_uds = (
            (1.0 + self._alphae) / 2.0 * self._phi[0:-1]
            + (1.0 - self._alphae) / 2.0 * self._phi[1:]
        )

        # Start QUICK face values from UDS values
        self._phie_quick[:] = self._phie_uds[:]

        # QUICK interpolation on interior faces only
        for i in range(1, self._grid.ncv):
            if self._Uhe[i] >= 0.0:
                self._phie_quick[i] = (
                    - self._phi[i - 1]
                    + 6.0 * self._phi[i]
                    + 3.0 * self._phi[i + 1]
                ) / 8.0
            else:
                self._phie_quick[i] = (
                    3.0 * self._phi[i]
                    + 6.0 * self._phi[i + 1]
                    - self._phi[i + 2]
                ) / 8.0

        # Face mass fluxes
        mdote = self._rho * self._Uhe * self._grid.Af

        # UDS fluxes
        flux_w_uds = self._const * mdote[:-1] * self._phie_uds[:-1]
        flux_e_uds = self._const * mdote[1:]  * self._phie_uds[1:]

        flux_uds = flux_e_uds - flux_w_uds

        # QUICK fluxes
        flux_w_quick = self._const * mdote[:-1] * self._phie_quick[:-1]
        flux_e_quick = self._const * mdote[1:]  * self._phie_quick[1:]

        flux_quick = flux_e_quick - flux_w_quick

        # Deferred correction:
        # implicit UDS + explicit QUICK correction
        flux = flux_uds + (flux_quick - flux_uds)

        # Mass imbalance term
        imbalance = (
            - self._const * mdote[1:]  * self._phi[1:-1]
            + self._const * mdote[:-1] * self._phi[1:-1]
        )

        # Linearization coefficients from UDS only
        coeffW = - self._const * mdote[:-1] * (1.0 + self._alphae[:-1]) / 2.0
        coeffE =   self._const * mdote[1:]  * (1.0 - self._alphae[1:])  / 2.0
        coeffP = - coeffW - coeffE

        # Boundary coefficient treatment
        coeffP[0] += coeffW[0] * self._west_bc.coeff()
        coeffP[-1] += coeffE[-1] * self._east_bc.coeff()

        coeffW[0] = 0.0
        coeffE[-1] = 0.0

        # Add to coefficient arrays
        coeffs.accumulate_aP(coeffP)
        coeffs.accumulate_aW(coeffW)
        coeffs.accumulate_aE(coeffE)
        coeffs.accumulate_rP(flux)
        coeffs.accumulate_rP(imbalance)

        return coeffs