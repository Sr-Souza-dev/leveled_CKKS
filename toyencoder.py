import numpy as np
from numpy.polynomial import Polynomial

# Encond complex vectors into polynomials
class CKKSToyEncoder:

    def __init__(self, M: int):
        """
            M should be a power of 2
            xi, which is an M-th root of unity. Will be used as a basis for our computations
            k-root = xi^k for gcd(k, M) = 1 (coprimes)
        """

        self.xi = np.exp(-2 * np.pi * 1j / M)
        self.M = M
        self.N = M // 2

    def vandermonde(self) -> np.array:
        # Computes the Vandermonde matrix from a m-th root of unity

        matrix = []

        for i in range (self.N):
            root = self.xi ** (2*i+1)   # It works because every odd number will be coprime with a power of 2
            row = []

            for j in range(self.N):
                row.append(root ** j)
            matrix.append(row)
        
        return matrix

    def sigma_inverse(self, b:np.array) -> Polynomial:
        # Encodes the vector b in a polynomial using an M-th root of unity (isomorphism)

        A = self.vandermonde()              # get the vandermode matrix from m-th root
        coeffs = np.linalg.solve(A, b)      # Solve the linear sistem (unique)
        poly = Polynomial(coeffs)           # Get a polynomial from the coefficients

        return poly
    
    def sigma(self, poly: Polynomial) -> np.array:
        # Decodes a polynomial by applying it to the M-th roots of unity

        outputs = []

        for i in range(self.N):
            root = self.xi ** (2*i+1)
            output = poly(root)
            outputs.append(output)
        
        return np.array(outputs)
