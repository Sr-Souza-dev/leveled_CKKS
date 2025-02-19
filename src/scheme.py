import numpy as np
from typing import Tuple, List
from src.encoder import CKKSEncoder
from src.params import LWEParams
from src.cryptogram import Cryptogram
from src.polynomial import PolynomialVec

class CkksScheme:
    params = LWEParams(
        decimal_precision=30,
        int_precision=30,
        N=16,
        sigma=3.2,
        level=3,  
    )
    encoder = CKKSEncoder(N=params.N, scale=params.scale)

    @classmethod
    def set_params(cls, params: LWEParams):
        """
        Set the parameters for the cryptogram
        """
        cls.params = params
        cls.encoder = CKKSEncoder(N=params.N, scale=params.scale)
        PolynomialVec.setClassProperty(params.N)

    @staticmethod
    def set_cryptogram_params(eval_keys: List[Tuple[PolynomialVec, PolynomialVec]]):
        Cryptogram.set_params(CkksScheme.params, eval_keys)

    @staticmethod
    def encrypt(z: np.array, sk: PolynomialVec) -> Cryptogram:

        m = CkksScheme.encoder.encode(z)
        a = CkksScheme.params.get_random_uniform_polynomial(CkksScheme.params.N, CkksScheme.params.qs[-1])
        e = CkksScheme.params.get_random_normal_polynomial(CkksScheme.params.N, CkksScheme.params.sigma)

        return Cryptogram(
            c0 = Cryptogram.reduce_polynomial((-a * sk) + e + m, CkksScheme.params.level),
            c1 = a,
            l = CkksScheme.params.level
        )

    @staticmethod
    def decrypt(c: Cryptogram, sk: PolynomialVec) -> np.array:
        c0, c1 = c.c
        p = Cryptogram.reduce_polynomial(c0 + c1 * sk, c.l)
        return CkksScheme.encoder.decode(p)

    
    

