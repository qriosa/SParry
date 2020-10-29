# pip install nvidia-ml-py3
from pynvml import *
from classes.device import Device

def deviceinfo():
	"""
	function: get infomation of the GPU device.
	
	parameters: None.
	
	return: return a device(class). (more info please see the developer documentation).
	"""

	# 实例化类
	device = device()
	return device