import numpy as np
from typing import Tuple, List
from src.params import LWEParams
from src.polynomial import PolynomialVec

class Cryptogram:
    params = LWEParams(
        decimal_precision=30,
        int_precision=30,
        N=16,
        sigma=3.2,
        level=3,  
    )
    eval_keys: List[Tuple[PolynomialVec, PolynomialVec]] = []

    def __init__(self,
        c0: PolynomialVec,         # b polynomial value
        c1: PolynomialVec,         # a polynomial value
        l: int,                 # level of the ciphertext
    ):
        self.c = (c0, c1)
        self.l = l


    @classmethod
    def set_params(cls, params: LWEParams, eval_keys: List[Tuple[PolynomialVec, PolynomialVec]]):
        """
        Set the parameters for the cryptogram
        """
        cls.params = params
        cls.eval_keys = eval_keys

    @staticmethod
    def reduce_polynomial(poly: PolynomialVec, level) -> PolynomialVec:
        return poly % Cryptogram.params.qs[level]
    
    @staticmethod
    def relinearize(d: Tuple[PolynomialVec, PolynomialVec, PolynomialVec], level) -> Tuple[PolynomialVec, PolynomialVec]:
        """
        Relinearize the ciphertext
        """
        d0, d1, d2 = d

        print("d0: ", d0)
        print("d1: ", d1)
        print("d2: ", d2)
        print("level: ", level)
        
        evk0, evk1 = Cryptogram.eval_keys[level]
        P0 = d0 + (evk0 * d2) // Cryptogram.params.p_scale
        P1 = d1 + (evk1 * d2) // Cryptogram.params.p_scale
        return (Cryptogram.reduce_polynomial(P0, level), Cryptogram.reduce_polynomial(P1, level))
    
    @staticmethod
    def rescale(c: Tuple[PolynomialVec, PolynomialVec], level) -> Tuple[PolynomialVec, PolynomialVec]:
        """
        Rescale the ciphertext
        """
        c0, c1 = c 
        c0 = c0 // Cryptogram.params.scale
        c1 = c1 // Cryptogram.params.scale

        return (Cryptogram.reduce_polynomial(c0, level), Cryptogram.reduce_polynomial(c1, level), level-1)

    def __mod__(self, other):
        if isinstance(other, int):
            return Cryptogram(
                c0= Cryptogram.reduce_polynomial(self.c[0], other),
                c1= Cryptogram.reduce_polynomial(self.c[1], other),
                l=self.l,
            )
        else:
            return NotImplemented

    def __add__(self, other):
        if isinstance(other, Cryptogram):
            return Cryptogram(
                c0 = self.c[0]+other.c[0], 
                c1 = self.c[1]+other.c[1],
                l = min(self.l, other.l),
            )
        elif isinstance(other, (int, float)):
            return Cryptogram(
                c0=self.c[0]+other, 
                c1=self.c[1]+other,
                l=self.l,
            )
        else:
            return NotImplemented
        
    def __mul__(self, other):
        if isinstance(other, Cryptogram):
            c0, c1 = self.c
            lv = min(self.l, other.l)

            # print("actual: ", self)
            # print("other: ", other)

            d0 = Cryptogram.reduce_polynomial(c0 * other.c[0], lv)
            d1 = Cryptogram.reduce_polynomial(c0 * other.c[1] + c1 * other.c[0], lv)
            d2 = Cryptogram.reduce_polynomial(c1 * other.c[1], lv)

            c0, c1 = Cryptogram.relinearize((d0, d1, d2), lv)
            c0, c1, lv = Cryptogram.rescale((c0, c1), lv)
            return Cryptogram(c0, c1, lv)
        
        elif isinstance(other, (int, float)):
            return Cryptogram(
                c0=self.c[0]*other, 
                c1=self.c[1]*other,
                l=self.l
            )
        else:
            return NotImplemented
        
        
    def __repr__ (self):
        return f"Cryptogram: \n\tc0: {self.c[0]}\n\tc1: {self.c[1]}\n\tLevel: {self.l}"