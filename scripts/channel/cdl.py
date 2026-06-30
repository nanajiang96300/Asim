"""CDL-B channel model using Kronecker spatial correlation.

Implements 3GPP TR 38.901 CDL-B parameters via the Kronecker model:
    H = R_RX^{1/2} · G · R_TX^{1/2}
where G is i.i.d. Rayleigh and R_TX, R_RX are spatial correlation matrices
built from the CDL-B angular spread parameters.
"""

import numpy as np


class CDLBChannel:
    """CDL-B MIMO channel (Kronecker model with 3GPP TR 38.901 parameters).

    3GPP CDL-B key parameters:
    - ASA (Azimuth Spread of Arrival): 10° at BS, 22° at UT
    - ZSA (Zenith Spread of Arrival): 7° at BS, 7° at UT
    - ASD (Azimuth Spread of Departure): 22° at BS, 10° at UT
    - ZSD (Zenith Spread of Departure): 3° at BS, 7° at UT

    For flat-fading MIMO we use the azimuth-only exponential correlation model:
        R[i,j] = rho^{|i-j|}  where rho = exp(-d/lambda * angle_spread^2 / 2)

    Correlation is controlled by angular spread: wider spread -> lower correlation.
    """

    # CDL-B typical values
    DEFAULT_ASA_DEG = 10.0   # Azimuth spread of arrival
    DEFAULT_ASD_DEG = 22.0   # Azimuth spread of departure

    def __init__(self,
                 asa_deg: float = DEFAULT_ASA_DEG,
                 asd_deg: float = DEFAULT_ASD_DEG,
                 wavelength: float = 0.5):
        """Initialize CDL-B channel generator.

        Args:
            asa_deg: azimuth spread of arrival in degrees
            asd_deg: azimuth spread of departure in degrees
            wavelength: element spacing in wavelengths (default 0.5 = lambda/2)
        """
        self.asa_deg = asa_deg
        self.asd_deg = asd_deg
        self.wavelength = wavelength
        # Convert angular spread to correlation coefficient
        # rho = exp(-(2*pi*d*sigma_angle)^2 / 2) where sigma is in radians
        sigma_asa = np.deg2rad(asa_deg)
        sigma_asd = np.deg2rad(asd_deg)
        self._rx_rho = np.exp(-0.5 * (2.0 * np.pi * wavelength * sigma_asa) ** 2)
        self._tx_rho = np.exp(-0.5 * (2.0 * np.pi * wavelength * sigma_asd) ** 2)

    @property
    def rx_correlation(self) -> float:
        """Receive-side (BS) per-element correlation coefficient."""
        return self._rx_rho

    @property
    def tx_correlation(self) -> float:
        """Transmit-side (UT) per-element correlation coefficient."""
        return self._tx_rho

    def _correlation_matrix(self, n: int, rho: float) -> np.ndarray:
        """Build exponential correlation matrix R[i,j] = rho^{|i-j|}."""
        indices = np.arange(n, dtype=np.float64)
        dist = np.abs(indices[:, None] - indices[None, :])
        return rho ** dist

    def _sqrtm(self, R: np.ndarray) -> np.ndarray:
        """Matrix square root via eigendecomposition: R^{1/2} = V·sqrt(D)·V^H."""
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals = np.maximum(eigvals, 0.0)
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T

    def generate(self, batch_size: int, nr: int, nt: int,
                 seed: int = None) -> np.ndarray:
        """Generate CDL-B MIMO channel matrix.

        Args:
            batch_size: number of independent channel realizations
            nr: number of receive antennas
            nt: number of transmit antennas
            seed: optional random seed

        Returns:
            H: complex channel matrix, shape (batch_size, nr, nt)
        """
        rng = np.random.RandomState(seed)

        R_rx = self._correlation_matrix(nr, self._rx_rho)
        R_tx = self._correlation_matrix(nt, self._tx_rho)
        R_rx_sqrt = self._sqrtm(R_rx)
        R_tx_sqrt = self._sqrtm(R_tx)

        H = np.zeros((batch_size, nr, nt), dtype=np.complex128)
        for b in range(batch_size):
            G_real = rng.randn(nr, nt).astype(np.float64)
            G_imag = rng.randn(nr, nt).astype(np.float64)
            G = (G_real + 1j * G_imag) / np.sqrt(2.0)
            H[b] = R_rx_sqrt @ G @ R_tx_sqrt

        if batch_size == 1:
            return H
        return np.array(H)
