
'''
return 0/1 sssp need/not need divide

return 2 apsp/msso not need
return 0/1 apsp/mssp need to be many sssp, and each sssp need divide or not
'''
from utils.debugger import Logger
from classes.device import Device
from utils.settings import forceDivide, noDivide

from math import sqrt
import numpy as np

logger = Logger(__name__)

def getPart(FREE, N):
    
    x = 0.1*FREE - 2*N
    
    return np.int32(x)


def judge_sssp(para):
    """
    function: 
        determine whether the current graph needs to use graph segmentation.
    
    parameters: 
        para: class, Parameter object. (see the 'sparry/classes/parameter.py/Parameter'). 
    
    return:
        bool, [0/1/2]. (more info please see the developer documentation).  
    """
    
    # this if is coming from APSP MSSP
    if para.device == None:
        # decide info
        para.device = Device()
        freeGpuMem = para.device.free
        logger.warning(f"SSSP: freeGpuMem = {freeGpuMem} Bytes = {freeGpuMem / 1024} KB = {freeGpuMem / 1024 / 1024} MB = {freeGpuMem / 1024 / 1024 / 1024} GB")
        
    device = para.device
    freeGpuMem = para.device.free

    # avoid overflow
    n = int(para.graph.n)
    m = int(para.graph.m)

    # can meet or not
    lowLimit = lambda PART, N, FREE: PART + (2*N) <= 0.1*FREE

    assert 0.1*freeGpuMem - 2*n > 0, "This algorithm is unable to solve problems of this scale on this device currently."

    # can put all m edges into GPU at once
    if lowLimit(m, n, device.free) and forceDivide == False:
        return 0

    if para.part != None:
        # over the max number is meanless
        para.part = np.int32(min(m, para.part))

        if lowLimit(para.part, n, device.free) == False:
            para.part = getPart(device.free, n)
            logger.warnning(f"your part is weak! I have corrected a new part = {para.part}")
    else:
        # if can put all edge, then put all edge into it.
        if lowLimit(m, n, device.free):
            para.part = m
            return 0

        else:
            # get part
            para.part = getPart(device.free, n) 
    
    logger.warning(f'part = {para.part}, m = {m}') 
    
    para.part = np.int32(para.part)

    return 1


def judge_mssp(para):
    """
    function: 
        determine whether the current graph needs to use graph segmentation.
    
    parameters: 
        para: class, Parameter object. (see the 'sparry/classes/parameter.py/Parameter').
    
    return:
        bool, [0/1/2]. (more info please see the developer documentation).  
    """
    
    # device info
    para.device = Device()
    freeGpuMem = para.device.free
    logger.warning(f"MSSP: freeGpuMem = {freeGpuMem} Bytes = {freeGpuMem / 1024} KB = {freeGpuMem / 1024 / 1024} MB = {freeGpuMem / 1024 / 1024 / 1024} GB")

    # the number of source vertices
    sNum = len(para.srclist)
    n = int(para.graph.n)
    m = int(para.graph.m)

    # don't need to divide
    if forceDivide == False and ((3 * sNum + 1) * n + 2 * m) <= 0.8 * freeGpuMem:
        return 2
    
    return judge_sssp(para)

def judge_apsp(para):
    """
    function: 
        determine whether the current graph needs to use graph segmentation.
    
    parameters: 
        para: class, Parameter object. (see the 'sparry/classes/parameter.py/Parameter').
    
    return:
        bool, [0/1/2]. (more info please see the developer documentation).    
    """


    # devide info
    para.device = Device()
    freeGpuMem = para.device.free

    logger.warning(f"APSP: freeGpuMem = {freeGpuMem} Bytes = {freeGpuMem / 1024} KB = {freeGpuMem / 1024 / 1024} MB = {freeGpuMem / 1024 / 1024 / 1024} GB")
    
    # no need
    if forceDivide == False:
        n = int(para.graph.n)
        m = int(para.graph.m)

        temp = (3 * n * n + 2 * n + 2 * m)

        if temp < 0.8 * freeGpuMem:
            return 2
    
    return judge_sssp(para)
 