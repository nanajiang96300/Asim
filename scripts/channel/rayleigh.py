"""Rayleigh fading channel: i.i.d. complex Gaussian H ~ CN(0, 1)."""

import numpy as np


class RayleighChannel:
    """i.i.d. Rayleigh fading MIMO channel.

    H[i,j] ~ CN(0, 1) = N(0, 1/2) + j * N(0, 1/2)
    """

    def __init__(self):
        pass

    def generate(self, batch_size: int, nr: int, nt: int,
                 seed: int = None) -> np.ndarray:
        """Generate H in C^{batch x nr x nt}.

        Args:
            batch_size: number of independent channel realizations
            nr: number of receive antennas
            nt: number of transmit antennas
            seed: optional random seed for reproducibility

        Returns:
            H: complex channel matrix, shape (batch_size, nr, nt)
        """
        rng = np.random.RandomState(seed)
        real = rng.randn(batch_size, nr, nt).astype(np.float32)
        imag = rng.randn(batch_size, nr, nt).astype(np.float32)
        return (real + 1j * imag) / np.sqrt(2.0)
