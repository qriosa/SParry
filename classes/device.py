# pip install nvidia-ml-py3
import pycuda.autoinit
import pycuda.driver as drv
from pynvml import *

class Device(object):
    """
    function: 
        get the GPU device infomation, get the type and attributes.

    parameters: 
        None, but 'self'.

    attributes:
        device: class, a pyCUDA device class object.
        CUDAVersion: str, the version of CUDA.
        driverVersion: int, the version of CUDA driver.
        deviceNum: int, the number of valid GPU device.
        deviceName: str, the name of the device.
        globalMem: int, the max number of the global memory.
        sharedMem: int, the max number of the shared memory.
        processNum: int, the number of processors.
        freeMem: int, the bytes of free memory.
        temperature: int, the temperature of the device.
        powerStstus: the power ststus of the device.

    method:
        get_device_type: get the type of the device.
        get_number_of_device: get the number of the device.
        get_version: obtain the version of CUDA against which PyCuda was compiled.
        get_driver_version: obtain the version of the CUDA driver on top of which PyCUDA is running. 
        getDeviceInfo: obtain the device infomation include: 'freeMemory, totalMemory, memoryUsed, temperature, powerStstus'.
        get_attributes: the pycuda's get_attributes.

    return 
        class, Result object. (see the 'SPoon/classes/device.py/Device').
    """
    def __init__(self):
        self.device = drv.Device(0) # device of pyCUDA

        self.CUDAVersion = None # compile CUDA version of pyCUDA 
        self.driverVersion = None # running driver version of CUDA 
        
        self.deviceName = None # gpu device model
        self.deviceNum = None # the number of CUDA device
        self.globalMem = None # capacity of global memory
        self.sharedMem = None # capacity of shared memory
        self.processNum = None # the number of processors

        self.SMNum = None # the number of SM
        self.gridSize = None # A tuple grid indicates that the two directions of the grid are the maximum values
        self.blockSize = None # The maximum value of the three directions of the same block

        self.total = None # The total capacity of video memory in GPU, in bytes
        self.free = None # Current total remaining in GPU
        self.used = None # video memory that be used
        self.temperature = None # temperature
        self.powerStstus = None # state of battery

        self.getDeviceInfo()

    def get_device_type(self):
        """
        function: 
            get the type of the device.
        
        parameters: 
            None but 'self'.
        
        return: 
            str, the type of the device.
        """

        if self.deviceName == None:
            self.deviceName = self.device.name()
        
        return self.deviceName

    def get_number_of_device(self):
        """
        function: 
            get the number of the device.
        
        parameters: 
            None but 'self'.
        
        return: 
            int, the number of the device.
        """

        if self.deviceNum == None:
            self.deviceNum = self.device.count()
        
        return self.deviceNum
    
    def get_version(self):
        """
        function: 
            obtain the version of CUDA against which PyCuda was compiled. 
        
        parameters: 
            None but 'self'.

        return:
            tuple, a 3-tuple of integers as (major, minor, revision).
        """
        if self.CUDAVersion == None:
            self.CUDAVersion = drv.get_version()
        
        return self.CUDAVersion
    
    def get_driver_version(self):
        """
        function: 
            obtain the version of the CUDA driver on top of which PyCUDA is running. 

        parameters: 
            None but 'self'.

        return:        
            int, an integer version number.
        """
        if self.driverVersion == None:
            self.driverVersion = drv.get_driver_version()
        
        return self.driverVersion
    
    def get_attributes(self):
        """
        function: 
            show the pycuda's get_attributes. 

        parameters: 
            None but 'self'.

        return:        
            str, the pycuda's get_attributes return.        
        """
        return self.device.get_attributes()

    def getDeviceInfo(self):
        """
        function: 
            obtain the device infomation include: 'freeMemory, totalMemory, memoryUsed, temperature, powerStstus'. 

        parameters: 
            None but 'self'.

        return:        
            None, no return.        
        """ 

        # init
        nvmlInit()

        # driver info
        nvmlSystemGetDriverVersion()

        # device name
        self.deviceNum = nvmlDeviceGetCount()
        self.deviceType = []

        for i in range(self.deviceNum):
            handle = nvmlDeviceGetHandleByIndex(i)
            self.deviceType.append(nvmlDeviceGetName(handle))

        # show the video memory, temperature, power, fan
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)

        # in Byte
        self.total = info.total
        self.free = info.free
        self.used = info.used

        self.temperature = nvmlDeviceGetTemperature(handle,0)
        self.powerStstus = nvmlDeviceGetPowerState(handle)

        # close
        nvmlShutdown()

