import numpy as np
from time import time

from utils.settings import INF
from classes.result import Result
from utils.debugger import Logger

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

logger = Logger(__name__)

def delta_stepping(para):
    """
    function: 
        use delta_stepping algorithm in GPU to solve the SSSP. 
    
    parameters:  
        class, Parameter object. (see the 'sparry/classes/parameter.py/Parameter'). 
    
    return: 
        class, Result object. (see the 'sparry/classes/result.py/Result').
    """

    logger.debug("turning to func delta_stepping-gpu-sssp")    
    
    with open('./method/sssp/cu/delta.cu', 'r', encoding = 'utf-8') as f:
        cuf = f.read()
    mod = SourceModule(cuf)

    # start time
    t1 = time()

    CSR, n, s, delta, pathRecordBool = para.graph.graph, para.graph.n, para.srclist, para.graph.delta, para.pathRecordBool
 
    if para.BLOCK != None:
        BLOCK = para.BLOCK
    else:
        BLOCK = (1024, 1, 1)
    
    if para.GRID != None:
        GRID = para.GRID
    else:
        GRID = (1, 1)

    V, E, W = CSR[0], CSR[1], CSR[2]

    s = np.int32(s)
    nowIsNull = np.int32(1)
    quickBreak = np.int32(1)
    
    # predist 
    predist = np.full((n, ), INF).astype(np.int32)

    # dist
    dist = np.full((n, ), INF).astype(np.int32)
    dist[s] = np.int32(0)

    # get function
    delta_sssp_cuda_fuc = mod.get_function("delta_stepping")

    # run!
    delta_sssp_cuda_fuc(drv.In(V), 
                        drv.In(E), 
                        drv.In(W), 
                        drv.In(n),
                        drv.In(s), 
                        drv.In(delta), 
                        drv.InOut(dist), 
                        drv.In(predist), 
                        drv.In(nowIsNull), 
                        drv.In(quickBreak), 
                        block = BLOCK, grid = GRID)

    timeCost = time() - t1
    
    # result
    result = Result(dist = dist, timeCost = timeCost, graph = para.graph)

    if pathRecordBool:
        result.calcPath()

    return result