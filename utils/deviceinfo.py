# pip install nvidia-ml-py3
from pynvml import *
from classes.device import Device

def deviceinfo():
    """
    function: 
        get infomation of the GPU device.
    
    parameters: 
        None.
    
    return: 
        class, Device object.(see the 'sparry/classes/device.py/Device').
    """

    # instantiation
    device = device()
    return device