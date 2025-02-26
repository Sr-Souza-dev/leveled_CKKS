from typing import Generic, TypeVar, Union
import numpy as np
from numpy.polynomial.polynomial import polymul


from mpmath import mp, mpf, mpc, pi, exp
from typing import List

mp.dps = 200  # Aumente conforme necessário para evitar perda de precisão

T = TypeVar("T", bound=np.generic)  # Garante que seja um tipo numérico compatível com numpy

class PolynomialVec(Generic[T]):
    N = 4

    def __init__(self, coeffs: Union[list, np.ndarray], dtype: T = np.int64):
        """
        Inicializa um polinômio representado como um vetor de coeficientes.

        :param coeffs: Lista ou np.array de coeficientes.
        :param dtype: Tipo dos coeficientes.
        """
        self.dtype = dtype  # Salva o tipo de coeficiente para uso interno
        self.coeffs = np.array(coeffs, dtype=self.dtype)  
        self.trim()

    def trim(self):
        """Remove coeficientes zero à direita para evitar polinômios com grau incorreto."""
        if len(self.coeffs) > 1:
            self.coeffs = np.trim_zeros(self.coeffs, 'b')

    @classmethod
    def setClassProperty(clc, N: int):
        clc.N = N

    @staticmethod
    def reduce_to_ring(coeffs: np.ndarray, q: int, dtype):
        # return coeffs
        """
        Reduz um polinômio arbitrário ao anel R = Z_q[X] / (X^N + 1).

        :param coeffs: np.array de coeficientes.
        :param q: Módulo para os coeficientes.
        :param dtype: Tipo numérico a ser usado.
        :return: np.array reduzido.
        """
        reduced_coeffs = np.zeros(PolynomialVec.N, dtype=dtype)

        for i, coef in enumerate(coeffs):
            if i < PolynomialVec.N:
                reduced_coeffs[i] += coef
            else:
                reduced_coeffs[i % PolynomialVec.N] += (-1) ** (i // PolynomialVec.N) * coef

        print("Received coeffs: ", coeffs)
        print("reduced_coeffs: ", reduced_coeffs)

        return PolynomialVec.centered_mod(reduced_coeffs, q, dtype)

    @staticmethod
    def centered_mod(coeffs: np.ndarray, q: int, dtype):
        """
        Aplica redução centrada módulo q para manter os coeficientes em [-q/2, q/2].

        :param coeffs: np.array - Coeficientes do polinômio.
        :param q: int - Módulo para redução.
        :param dtype: Tipo numérico a ser usado.
        :return: np.array - Coeficientes reduzidos centrados.
        """
        coeffs = np.mod(coeffs, q).astype(dtype)
        coeffs[coeffs > q // 2] -= q  
        return coeffs

    def __repr__(self):
        """Representação do polinômio na forma de string."""
        return str(self.coeffs)

    def __call__(self, x: float) -> float:
        """Avalia o polinômio em um dado valor x usando a Regra de Horner."""
        result = self.dtype(0)
        for coef in reversed(self.coeffs):
            result = result * x + coef
        return result

    def __add__(self, other: Union[int, "PolynomialVec[T]"]):
        """Soma dois polinômios ou um polinômio e um escalar."""
        if isinstance(other, int):
            return PolynomialVec(self.coeffs + other, self.dtype)
        elif isinstance(other, PolynomialVec):
            max_len = max(len(self.coeffs), len(other.coeffs))
            new_coeffs = np.zeros(max_len, dtype=self.dtype)

            new_coeffs[:len(self.coeffs)] += self.coeffs
            new_coeffs[:len(other.coeffs)] += other.coeffs

            return PolynomialVec(new_coeffs, self.dtype)
        else:
            raise TypeError("A soma deve ser feita entre dois polinômios ou um polinômio e um escalar.")

    def __sub__(self, other: Union[int, "PolynomialVec[T]"]):
        """Subtrai dois polinômios."""
        if isinstance(other, int):
            return PolynomialVec(self.coeffs - other, self.dtype)
        elif isinstance(other, PolynomialVec):
            max_len = max(len(self.coeffs), len(other.coeffs))
            new_coeffs = np.zeros(max_len, dtype=self.dtype)
            new_coeffs[:len(self.coeffs)] += self.coeffs
            new_coeffs[:len(other.coeffs)] -= other.coeffs
            return PolynomialVec(new_coeffs, self.dtype)
        else:
            raise TypeError("A subtração deve ser feita entre dois polinômios ou um polinômio e um escalar.")
        
    def __truediv__(self, other):
        """Divisão polinomial (longa divisão de polinômios)."""
        if isinstance(other, PolynomialVec):
            return self.polynomial_division(other)
        elif isinstance(other, (int)):
            return PolynomialVec(self.coeffs / other, dtype=np.float64)  
        else:
            raise TypeError("A divisão deve ser feita entre dois polinômios.")


    def __neg__(self):
        """Inverte o sinal do polinômio (-p)."""
        return PolynomialVec(-self.coeffs, self.dtype)

    def __mul__(self, other: Union[int, "PolynomialVec[T]"]):
        """Multiplica dois polinômios usando FFT."""
        if isinstance(other, int):
            return PolynomialVec(self.coeffs * other, self.dtype)
        elif isinstance(other, PolynomialVec):
                    """Multiplica polinômios usando DFT/IDFT implementadas manualmente."""
        if isinstance(other, PolynomialVec):
            # Configuração de precisão
            original_dps = mp.dps
            mp.dps = 100  # Precisão suficiente para coeficientes grandes
            
            len_res = len(self.coeffs) + len(other.coeffs) - 1
            next_pow2 = 2 ** int(np.ceil(np.log2(len_res)))
            
            a = self._pad_coeffs(next_pow2)
            b = other._pad_coeffs(next_pow2)
            
            dft_a = self._dft(a)
            dft_b = self._dft(b)
            product_dft = [x * y for x, y in zip(dft_a, dft_b)]
            
            result = self._idft(product_dft)
            final_coeffs = [self._safe_int(x.real) for x in result[:len_res]]
            
            mp.dps = original_dps
            return PolynomialVec(final_coeffs, dtype=self.dtype)

            # len_res = len(self.coeffs) + len(other.coeffs) - 1
            # next_pow2 = 2 ** int(np.ceil(np.log2(len_res)))

            # f1 = np.fft.rfft(self.coeffs, next_pow2)
            # f2 = np.fft.rfft(other.coeffs, next_pow2)

            # result_freq = f1 * f2
            # result_coeffs = np.fft.irfft(result_freq).real.round().astype(self.dtype)

            # return PolynomialVec(result_coeffs[:len_res], self.dtype)
        else:
            raise TypeError("A multiplicação deve ser feita entre dois polinômios ou um polinômio e um escalar.")

    def __floordiv__(self, scalar: int):
        """Divisão inteira de um polinômio por um número inteiro."""
        if not isinstance(scalar, int):
            raise TypeError("A divisão inteira só pode ser feita com um número inteiro.")
        return PolynomialVec(self.coeffs // scalar, self.dtype)
    
    def _pad_coeffs(self, target_len: int) -> List[mpf]:
        pad_length = target_len - len(self.coeffs)
        padded = list(self.coeffs) + [0] * pad_length
        return [mpf(int(x)) for x in padded]

    @staticmethod
    def _dft(signal: List[mpf]) -> List[mpc]:
        N = len(signal)
        return [
            sum(mpf(x) * exp(-2j * pi * k * n / N) 
            for n, x in enumerate(signal))
            for k in range(N)
        ]

    @staticmethod
    def _idft(freq: List[mpc]) -> List[mpc]:
        N = len(freq)
        return [
            sum(X * exp(2j * pi * k * n / N) 
            for k, X in enumerate(freq)) / N
            for n in range(N)
        ]

    @staticmethod
    def _safe_int(value: mpf) -> int:
        """Converte mpf para int sem overflow."""
        return int(mp.nstr(value, mp.dps, strip_zeros=False).split('.')[0])


    def __mod__(self, q: int):
        """Reduz os coeficientes módulo q."""
        return PolynomialVec(PolynomialVec.reduce_to_ring(self.coeffs, q, self.dtype), self.dtype)