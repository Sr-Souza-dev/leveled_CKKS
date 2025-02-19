import numpy as np

class PolynomialVec:
    N = 16
    def __init__(self, coeffs: np.array):
        """
        Inicializa um polinômio representado como um vetor de coeficientes.

        :param coeffs: Lista ou np.array de coeficientes inteiros.
        """
        self.coeffs = np.array(coeffs, dtype=np.int64)  # Garante que os coeficientes são np.int64
        self.trim()
    
    def trim(self):
        """Remove coeficientes zero à direita para evitar polinômios com grau incorreto."""
        if len(self.coeffs) > 1:
            self.coeffs = np.trim_zeros(self.coeffs, 'a')

    @classmethod
    def setClassProperty(clc, N: int):
        clc.N = N

    @staticmethod
    def reduce_to_ring(coeffs: np.array, q: int) -> np.array:
        """
        Reduz um polinômio arbitrário ao anel R = Z_q[X] / (X^N + 1).

        :param poly: PolynomialVec - Polinômio a ser reduzido.
        :param N: int - Grau do anel (x^N + 1).
        :param q: int - Módulo para os coeficientes.
        :return: PolynomialVec - Polinômio reduzido em R.
        """

        reduced_coeffs = np.zeros(PolynomialVec.N, dtype=np.int64)  # Inicializa vetor de coeficientes reduzido

        for i, coef in enumerate(coeffs):
            if i < PolynomialVec.N:
                reduced_coeffs[i] += coef
            else:
                # Redução pelo polinômio X^N ≡ -1 (equivalente a X^N = -1 mod q)
                reduced_coeffs[i % PolynomialVec.N] += (-1) ** (i // PolynomialVec.N) * coef

        # Aplica a redução centrada para manter coeficientes em [-q/2, q/2]
        reduced_coeffs = PolynomialVec.centered_mod(reduced_coeffs, q)

        return reduced_coeffs

    @staticmethod
    def centered_mod(coeffs, q):
        """
        Aplica redução centrada módulo q para manter os coeficientes em [-q/2, q/2].

        :param coeffs: np.array - Coeficientes do polinômio.
        :param q: int - Módulo para redução.
        :return: np.array - Coeficientes reduzidos centrados.
        """
        coeffs = np.mod(coeffs, q)
        coeffs[coeffs > q // 2] -= q  # Converte para o intervalo [-q/2, q/2]
        return coeffs
    
    
    def __repr__(self):
        """
        Representação do polinômio na forma de string.
        """
        return str(self.coeffs)

    def __add__(self, other):
        """
        Soma dois polinômios.
        
        :param other: PolynomialVec
        :return: PolynomialVec com os coeficientes somados
        """
        if(isinstance(other, (int))):
            return PolynomialVec(self.coeffs + other)
        elif isinstance(other, PolynomialVec):
            max_len = max(len(self.coeffs), len(other.coeffs))
            new_coeffs = np.zeros(max_len, dtype=np.int64)

            # Soma os coeficientes correspondentes
            new_coeffs[:len(self.coeffs)] += self.coeffs
            new_coeffs[:len(other.coeffs)] += other.coeffs

            return PolynomialVec(new_coeffs)
        else:
            raise TypeError("A soma deve ser feita entre dois polinômios ou um polinômio e um escalar.")
    
    def __sub__(self, other):
        """Subtrai dois polinômios."""
        if isinstance(other, (int)):
            return PolynomialVec(self.coeffs - other)
        elif isinstance(other, PolynomialVec):
            max_len = max(len(self.coeffs), len(other.coeffs))
            new_coeffs = np.zeros(max_len, dtype=np.int64)
            new_coeffs[:len(self.coeffs)] += self.coeffs
            new_coeffs[:len(other.coeffs)] -= other.coeffs
            return PolynomialVec(new_coeffs)
        else:
            raise TypeError("A subtração deve ser feita entre dois polinômios ou um polinômio e um escalar")

    def __neg__(self):
        """Inverte o sinal do polinômio (-p)."""
        return PolynomialVec(-self.coeffs)


    def __mul__(self, other):
        """
        Multiplica dois polinômios usando FFT (DFT Rápida).
        
        :param other: PolynomialVec
        :return: PolynomialVec resultante da multiplicação
        """

        if isinstance(other, (int)):
            return PolynomialVec(self.coeffs * other)
        elif isinstance(other, PolynomialVec):
            len_res = len(self.coeffs) + len(other.coeffs) - 1
            next_pow2 = 2 ** int(np.ceil(np.log2(len_res)))  # Encontra o próximo tamanho potência de 2

            # Expande os coeficientes para tamanho adequado para a FFT
            f1 = np.fft.rfft(self.coeffs, next_pow2)
            f2 = np.fft.rfft(other.coeffs, next_pow2)

            # Multiplicação no domínio da frequência
            result_freq = f1 * f2

            # Transformada inversa para voltar ao domínio do tempo
            result_coeffs = np.fft.irfft(result_freq).real.round().astype(np.int64)

            # Redimensiona para o tamanho correto
            return PolynomialVec(result_coeffs[:len_res])
        else: 
            raise TypeError("A multiplicação deve ser feita entre dois polinômios ou um polinômio e um escalar.")
    
    def __floordiv__(self, scalar):
        """Divisão inteira de um polinômio por um número inteiro."""
        if not isinstance(scalar, (int, np.integer)):
            raise TypeError("A divisão inteira só pode ser feita com um número inteiro.")
        return PolynomialVec(self.coeffs // scalar)

    def __truediv__(self, other):
        """Divisão polinomial (longa divisão de polinômios)."""
        if isinstance(other, PolynomialVec):
            return self.polynomial_division(other)
        elif isinstance(other, (int)):
            return PolynomialVec(self.coeffs // other)
        else:
            raise TypeError("A divisão deve ser feita entre dois polinômios.")

    def polynomial_division(self, divisor):
        """
        Realiza a divisão longa de polinômios: self / divisor.

        :param divisor: PolynomialVec - Polinômio divisor.
        :return: (quociente, resto) - Tuple[PolynomialVec, PolynomialVec]
        """
        if len(divisor.coeffs) == 0 or np.all(divisor.coeffs == 0):
            raise ZeroDivisionError("Divisão por polinômio zero não é permitida.")

        dividend = np.copy(self.coeffs).astype(np.int64)
        divisor_coeffs = np.copy(divisor.coeffs).astype(np.int64)
        quotient = np.zeros(len(dividend) - len(divisor_coeffs) + 1, dtype=np.int64)

        for i in range(len(quotient) - 1, -1, -1):
            quotient[i] = dividend[i + len(divisor_coeffs) - 1] // divisor_coeffs[-1]
            dividend[i:i + len(divisor_coeffs)] -= quotient[i] * divisor_coeffs

        remainder = np.trim_zeros(dividend, 'b')

        return PolynomialVec(quotient), PolynomialVec(remainder)

    def __mod__(self, q):
        """
        Reduz os coeficientes módulo q.
        
        :param q: Inteiro módulo.
        :return: PolynomialVec com coeficientes reduzidos.
        """
        return PolynomialVec(PolynomialVec.reduce_to_ring(self.coeffs, q))
