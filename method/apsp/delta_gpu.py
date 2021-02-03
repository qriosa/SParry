from time import time
import numpy as np

from classes.result import Result
from utils.settings import INF
from utils.debugger import Logger

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

logger = Logger(__name__)

def delta_stepping(para):
    """
    function: 
        use delta_stepping algorithm in GPU to solve the APSP. 
    
    parameters:  
        class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter').
    
    return: 
        class, Result object. (see the 'SPoon/classes/result.py/Result').
    """

    with open('./method/apsp/cu/delta.cu', 'r', encoding = 'utf-8') as f:
        cuf = f.read()
    mod = SourceModule(cuf)

    logger.debug("turning to func delta_stepping-gpu-apsp")

    # start time
    t1 = time()

    CSR, n, delta, pathRecordBool = para.graph.graph, para.graph.n, para.graph.delta, para.pathRecordBool

    V, E, W = CSR[0], CSR[1], CSR[2]

    # global parameters of block and grid 
    if para.BLOCK != None:
        BLOCK = para.BLOCK
    else:
        BLOCK = (1024, 1, 1)
    
    if para.GRID != None:
        GRID = para.GRID
    else:
        GRID = (512, 1)

    predist = np.full((n * n, ), INF).astype(np.int32)
    dist = np.full((n * n, ), INF).astype(np.int32)
    B = np.full((n * GRID[0], ), -1).astype(np.int32)
    hadin = np.full((n * GRID[0], ), 0).astype(np.int32)

    # init all source vertex
    for i in range(n):
        # i is the source vertex 
        dist[i * n + i] = np.int32(0)

    # get function
    delta_apsp_cuda_fuc = mod.get_function("delta_stepping")

    # run!
    delta_apsp_cuda_fuc(drv.In(V),
                        drv.In(E),
                        drv.In(W),
                        drv.In(n),
                        drv.In(delta),
                        drv.InOut(dist),
                        drv.In(predist),
                        drv.In(B),
                        drv.In(hadin),
                        block = BLOCK,
                        grid = GRID)

    timeCost = time() - t1
    
    # result
    result = Result(dist = dist, timeCost = timeCost, graph = para.graph)

    if pathRecordBool:
        result.calcPath()

    return result
    