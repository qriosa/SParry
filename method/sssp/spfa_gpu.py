#-*-coding:utf-8-*-
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import time
import random
import os

from classes.result import Result
from utils.settings import INF
from utils.debugger import Logger

kernelName='spfaKernelForSSSP'
fr=open("./method/sssp/cu/spfa.cu","r",encoding='utf-8')
kernelCode=fr.read()
fr.close()
mod=SourceModule(kernelCode)
KERNEL=mod.get_function(kernelName)

logger = Logger(__name__)

def spfa(para):
    """
    function: 
        use spfa algorithm in GPU to solve the APSP. 
    
    parameters:  
        class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter').
    
    return: 
        class, Result object. (see the 'SPoon/classes/result.py/Result').
    """

    logger.debug("turning to func spfa-gpu-sssp")

    CSR = para.graph.graph
    n = para.graph.n 
    st = para.srclist
    pathRecording = para.pathRecordBool

    if(para.GRID is not None):
        GRID = para.GRID
    else:
        GRID=(1,1,1)
    if(para.BLOCK is not None):
        BLOCK = para.BLOCK
    else:
        BLOCK=(1024,1,1)

    start_time = time.process_time()
    RST_V_np32 = CSR[0]
    RST_E_np32 = CSR[1]
    RST_W_np32 = CSR[2]
    MAP_N_np32 = n
    
    VISIT = [0 for i in range(n)]
    VISIT[st] = 1
    VISIT_np32  = np.copy(VISIT).astype(np.int32)

    DIST = [INF for i in range(n)]
    DIST[st] = 0
    DIST_np32  = np.copy(DIST).astype(np.int32)

    KERNEL(
        drv.In(RST_V_np32), drv.In(RST_E_np32), drv.In(RST_W_np32),
        drv.In(MAP_N_np32), drv.In(VISIT_np32), drv.InOut(DIST_np32),
        block=BLOCK, grid=GRID
    )
    end_time=time.process_time()
    timeCost = end_time - start_time
    result = Result(dist = DIST_np32, timeCost = timeCost, graph = para.graph)
    
    if(pathRecording):
        result.calcPath()

    return result

    