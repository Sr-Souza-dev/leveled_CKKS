import numpy as np
from numpy.polynomial import Polynomial
from typing import Tuple, List
from src.encoder import CKKSEncoder

class CkksScheme:

    def __init__(self, 
                 encoder: CKKSEncoder, 
                 secret_key: Polynomial, 
                 noise_scale: float,
                 p: int,
                 qs: List[int]):
        """
        :param encoder: Instance of the CKKSEncoder class
        :param secret_key: Polynomial representing the secret key
        :param noise_scale: Scale factor for noise addition
        :param p: Factor used in relinearization
        :param qs: List of moduli for different levels
        """
        self.p = p
        self.qs = qs  # Array de moduli para diferentes níveis
        self.q_index = len(qs) - 1  # Começa no nível mais alto
        self.q = qs[self.q_index]  # Modulus inicial
        self.encoder = encoder
        self.secret_key = secret_key
        self.noise_scale = noise_scale
        self.public_key = self.generate_public_key()
        self.eval_key = self.generate_relinearization_key()

    def generate_public_key(self) -> Tuple[Polynomial, Polynomial]:
        a = Polynomial(np.random.uniform(-1, 1, self.encoder.N).astype(np.float64))
        e = Polynomial(np.random.normal(0, self.noise_scale, self.encoder.N).astype(np.float64))
        pk0 = self.reduce_polynomial(-self.secret_key * a + e) 
        pk1 = a 
        return pk0, pk1

    def generate_relinearization_key(self) -> Tuple[Polynomial, Polynomial]:
        a0 = Polynomial(np.random.uniform(-1, 1, self.encoder.N).astype(np.float64))
        e0 = Polynomial(np.random.normal(0, self.noise_scale, self.encoder.N).astype(np.float64))
        evk0 = self.reduce_polynomial(-a0 * self.secret_key + e0 + self.p * (self.secret_key**2)) 
        evk1 = a0 
        return evk0, evk1

    def reduce_polynomial(self, poly: Polynomial) -> Polynomial:
        # return poly;
        coeffs_mod_q = np.mod(poly.coef, self.q)
        reduced_coeffs = np.zeros(self.encoder.N)
        for i, coef in enumerate(coeffs_mod_q):
            if i < self.encoder.N:
                reduced_coeffs[i] += coef
            else:
                reduced_coeffs[i % self.encoder.N] += (-1)**(i // self.encoder.N) * coef
        reduced_coeffs = np.mod(reduced_coeffs, self.q)
        return Polynomial(reduced_coeffs)

    def encrypt(self, z: np.array) -> Tuple[Polynomial, Polynomial]:
        m = self.encoder.encode(z)
        pk0, pk1 = self.public_key
        e = Polynomial(np.random.normal(0, self.noise_scale, self.encoder.N).astype(np.float64))
        u = Polynomial(np.random.uniform(-1, 1, self.encoder.N).astype(np.float64))
        c0 = self.reduce_polynomial(pk0 * u + m + e) 
        c1 = self.reduce_polynomial(pk1 * u + e) 
        return c0, c1

    def decrypt(self, ciphertext: Tuple[Polynomial, Polynomial]) -> np.array:
        c0, c1 = ciphertext
        m_approx = self.reduce_polynomial(c0 + self.secret_key * c1) 
        return self.encoder.decode(m_approx)

    def add(self, c1: Tuple[Polynomial, Polynomial], c2: Tuple[Polynomial, Polynomial]):
        return tuple(self.reduce_polynomial(a + b) for a, b in zip(c1, c2))
    
    @staticmethod
    def mult_const(c: Tuple[Polynomial, Polynomial], val: float):
        return tuple(a * val for a in c)

    def mult(self, c1: Tuple[Polynomial, Polynomial], c2: Tuple[Polynomial, Polynomial]) -> Tuple[Polynomial, Polynomial]:
        c1_0, c1_1 = c1
        c2_0, c2_1 = c2

        d0 = self.reduce_polynomial(c1_0 * c2_0) 
        d1 = self.reduce_polynomial(c1_0 * c2_1 + c1_1 * c2_0) 
        d2 = self.reduce_polynomial(c1_1 * c2_1) 

        c_lin = self.relinearization(d0, d1, d2)
        c_rescaled = self.rescale(c_lin)
        return c_rescaled
    
    def relinearization(self, d0: Polynomial, d1: Polynomial, d2: Polynomial) -> Tuple[Polynomial, Polynomial]:
        # Relinearization
        evk0, evk1 = self.eval_key
        p_term = (np.floor((d2 * evk0).coef / self.p), np.floor((d2 * evk1).coef))
        relin_c0 = self.reduce_polynomial(d0 + Polynomial(p_term[0]))
        relin_c1 = self.reduce_polynomial(d1 + Polynomial(p_term[1]))

        return relin_c0, relin_c1

    def rescale(self, ciphertext: Tuple[Polynomial, Polynomial]) -> Tuple[Polynomial, Polynomial]:
        # return ciphertext
        """
        Perform rescaling to reduce the modulus from q_l to q_{l-1}.
        """
        if self.q_index == 0:
            raise ValueError("Cannot rescale further, already at the lowest level.")
        
        c0, c1 = ciphertext
        q_l = self.qs[self.q_index]
        q_l_minus_1 = self.qs[self.q_index - 1]
        scale_factor = q_l_minus_1 / q_l

        # Update current modulus
        self.q_index -= 1
        self.q = self.qs[self.q_index]

        # Rescaling both parts of the ciphertext
        c0_rescaled = np.round(scale_factor * c0.coef)
        c1_rescaled = np.round(scale_factor * c1.coef) 


        return self.reduce_polynomial(Polynomial(c0_rescaled)), self.reduce_polynomial(Polynomial(c1_rescaled))
