def int_divide_list_by_scalar(lst, scalar):
    """
    Divide todos os elementos de uma lista por um escalar.

    :param lst: List - Lista de elementos.
    :param scalar: int - Escalar para divisão.
    :return: List - Lista com elementos divididos.
    """
    return [x // scalar for x in lst]

def divide_list_by_scalar(lst, scalar):
    """
    Divide todos os elementos de uma lista por um escalar.

    :param lst: List - Lista de elementos.
    :param
    scalar: int - Escalar para divisão.
    :return: List - Lista com elementos divididos.
    """
    return [x / scalar for x in lst]

def multiply_list_by_scalar(lst, scalar):
    """
    Multiplica todos os elementos de uma lista por um escalar.

    :param lst: List - Lista de elementos.
    :param scalar: int - Escalar para multiplicação.
    :return: List - Lista com elementos multiplicados.
    """
    return [x * scalar for x in lst]

def add_lists(lst1, lst2):
    """
    Soma os elementos de duas listas.

    :param lst1: List - Primeira
    :param lst2: List - Segunda
    :return: List - Lista com elementos somados.
    """
    max_len = max(len(lst1), len(lst2))
    new_lst = [0] * max_len

    # Soma os coeficientes correspondentes
    new_lst[:len(lst1)] += lst1
    new_lst[:len(lst2)] += lst2

    return new_lst
    
def mod_list(lst, q):
    """
    Reduz os coeficientes de uma lista módulo q.

    :param lst: List - Lista de coeficientes.
    :param q: int - Módulo para redução.
    :return: List - Lista com coeficientes reduzidos.
    """
    return [x % q for x in lst]

def round_list(lst):
    """
    Arredonda os elementos de uma lista.

    :param lst: List - Lista de elementos.
    :return: List - Lista com elementos arredondados.
    """
    return [round(x) for x in lst]
