
import numpy as np
from numpy.polynomial import Polynomial

# Encond complex vectors into polynomials
class CKKSEncoder:

    def __init__(self, M: int, scale: float):
        """
            M should be a power of 2
            xi, which is an M-th root of unity. Will be used as a basis for our computations
            k-root = xi^k for gcd(k, M) = 1 (coprimes)
        """

        self.xi = np.exp(-2 * np.pi * 1j / M)
        self.M = M
        self.N = M // 2
        self.scale = scale
        self.create_sigma_R_basis()

        vec = np.zeros(self.N+1, dtype=int)
        vec[0] = 1
        vec[-1] = 1
        self.module = Polynomial(vec)

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
    
    def pi(self, z: np.array) -> np.array:
        # Projects a vector of H (C^N) into C^{N/2} (therefore M/4)

        K = self.N // 2
        return z[:K]
    
    @staticmethod
    def pi_inverse(z: np.array) -> np.array:
        # Expands a vector of C^{N/2} by expanding it with its complex conjugate in H (C^N)

        z_conjugate = z[::-1]           # Invert the array
        z_conjugate = [np.conjugate(x) for x in z_conjugate]
        return np.concatenate([z, z_conjugate])
    
    def create_sigma_R_basis(self): 
        # Create the basis of sigma (sigma(1), sigma(X), ..., sigma(X**{N-1})) with respect to the cyclotomic polynomial basis
        self.sigma_R_basis = np.array(self.vandermonde()).T

    def compute_basis_coordinates(self, z: np.array) -> np.array:
        # Computes the coordinates of a vector with respect to the orthogonal lattice basis

        coordinates = np.array([np.real(np.vdot(z, b) / np.vdot(b, b)) for b in self.sigma_R_basis])
        return coordinates
    
    @staticmethod
    def integral_rest(coordinates: np.array):
        # Gives the integral rest
        coordinates = coordinates - np.floor(coordinates)
        return coordinates
    
    def coordinate_wise_random_rouding(self, coordinates: np.array) -> np.array:
        # Rounds coordinates randonmly
        rest = self.integral_rest(coordinates)
        rand = np.array([np.random.choice([r, r-1], 1, [1-r, r]) for r in rest]).reshape(-1)

        round = coordinates - rand
        round = [int(coeff) for coeff in round]
        return round

    def sigma_R_discretization(self, z: np.array) -> np.array:
        # Projects a vector on the lattice using coordinate wise random rounding

        coordinates = self.compute_basis_coordinates(z)

        rounded_coordinates = self.coordinate_wise_random_rouding(coordinates)
        proj = np.matmul(self.sigma_R_basis.T, rounded_coordinates)
        return proj
    
    def encode(self, z: np.array) -> Polynomial:
        """
            Encondes a vector (C^{N/2}) by expanding it first to H, 
            scale it, project it on the lattice of sigma(R), 
            and performs sigma inverse.
        """
        hz = self.pi_inverse(z)
        scaled_hz = self.scale * hz
        rounded_scaled_hz = self.sigma_R_discretization(scaled_hz)
        coeff = self.sigma_inverse(rounded_scaled_hz)

        # Round it afterwards due to numerical imprecision
        coeff = np.round(np.real(coeff.coef)).astype(int)
        poly = Polynomial(coeff)
        return poly

    def decode(self, poly: Polynomial) -> np.array:
        # Decodes a polynomial by removing the scale, evaluating on roots, and project it on C^{N/2}

        poly = poly % self.module
        rescaled_p = poly / self.scale
        z = self.sigma(rescaled_p)
        z = self.pi(z)
        return z


