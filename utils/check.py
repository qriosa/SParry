import hashlib
import numpy as np

from utils.myPrint import PRINT_blue
from utils.myPrint import PRINT_red

def check(data1, data2, name1 = 'data1', name2 = 'data2'):
    """
    function: 
        to check the data is equal or not.

    parameters:
        data1: numpy.ndarray, must, first data.
        data2: numpy.ndarray, must, second data.
        name1: str, the name of data1.
        name2: str, the name of data2.
    
    return: 
        str, the EQUAL or NOT. 
    """
    m1 = hashlib.md5()
    m2 = hashlib.md5()

    m1.update(str(np.array(data1).flatten()).encode('utf-8'))
    m2.update(str(np.array(data2).flatten()).encode('utf-8'))

    if m1.hexdigest() == m2.hexdigest():
        res = f"{name1} and {name2} are EQUAL"
        PRINT_blue(res)
        return res
    else:
        res = f"{name1} and {name2} are NOT EQUAL"
        PRINT_red(res)
        return res



def checkBool(data1, data2):
    """
    function: 
        to check the data is equal or not.

    parameters:
        data1: numpy.ndarray, must, first data.
        data2: numpy.ndarray, must, second data.
    
    return: 
        bool, True/False.
    """
    m1 = hashlib.md5()
    m2 = hashlib.md5()

    m1.update(str(np.array(data1).flatten()).encode('utf-8'))
    m2.update(str(np.array(data2).flatten()).encode('utf-8'))

    if m1.hexdigest() == m2.hexdigest():
        return True
    else:
        return False