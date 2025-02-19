import numpy as np
from src.polynomial import PolynomialVec


class LWEParams:

    def __init__(self,
        N: int, 
        int_precision: int, 
        decimal_precision: int,
        sigma: float,
        level: int):

        """
        :param N: Polynomial degree (number of coefficients/slots)
        :param int_precision: Number of integer bits
        :param decimal_precision: Number of decimal bits
        :param sigma: Standard deviation for noise generation
        :param level: Number of supported multiplicative levels
        """

        self.N = N
        self.sigma = sigma
        self.level = level
        self.scale = 2**decimal_precision

        qs = [self.scale * (2**int_precision)]

        for i in range(level):
            qs.append(qs[-1]*self.scale)

        self.qs = qs
        self.p_scale = qs[0]**2

    @staticmethod
    def get_random_uniform_polynomial(N: int, desvio: float) -> PolynomialVec:
        return PolynomialVec(np.random.randint(-desvio//2, desvio//2, N))
    
    @staticmethod
    def get_random_normal_polynomial(N: int, desvio: float) -> PolynomialVec:
        return PolynomialVec(np.random.normal(0, desvio, N).astype(np.int64))
    
    def generate_sk(self):
        return LWEParams.get_random_uniform_polynomial(self.N, self.qs[0])
    
    def generate_pk(self, sk: PolynomialVec):
        a = LWEParams.get_random_uniform_polynomial(self.N, self.qs[0])
        e = LWEParams.get_random_normal_polynomial(self.N, self.sigma)
        pk0:PolynomialVec = -sk * a + e 
        pk1:PolynomialVec = a 
        return (pk0, pk1)
    
    def generate_eval_keys(self, sk: PolynomialVec):
        eval_keys = [None]
        for i in range(self.level):
            a = LWEParams.get_random_uniform_polynomial(self.N, self.qs[i+1])
            e = LWEParams.get_random_normal_polynomial(self.N, self.sigma)
            evk0 = -a * sk + e + (sk*sk) * self.p_scale
            evk1 = a 
            eval_keys.append((evk0, evk1))
        
        return eval_keys
    
    def generate_keys(self):
        sk = self.generate_sk()
        pk = self.generate_pk(sk)
        eval_keys = self.generate_eval_keys(sk)

        return sk, pk, eval_keys

    def __str__(self):
        return f"LWEParams(N={self.N}, sigma={self.sigma}, level={self.level}) \nscale={self.scale} \np_scale={self.p_scale} \nqs={self.qs},"

