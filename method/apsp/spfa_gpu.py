import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import time
import numpy as np
import random
import os

from classes.result import Result
from utils.settings import INF
from utils.debugger import Logger

kernelName="kernelForAPSP"
fr=open("./method/apsp/cu/spfa.cu","r",encoding='utf-8')
kernelCode=fr.read()
fr.close()
mod=SourceModule(kernelCode)
KERNEL=mod.get_function(kernelName)
BLOCK=(1024,1,1)
GRID=(512,1)

logger = Logger(__name__)

def spfa(para):
    """
    function: 
        use spfa algorithm in GPU to solve the APSP. 
    
    parameters:  
        class, Parameter object.
    
    return: 
        class, Result object. (more info please see the developer documentation) .
    """

    logger.info("turning to func spfa-gpu-apsp")

    CSR = para.CSR
    n = para.n 
    pathRecording = para.pathRecordBool

    if(para.GRID is not None):
        GRID = para.GRID
    else:
        GRID=(512,1,1)
    if(para.BLOCK is not None):
        BLOCK = para.BLOCK
    else:
        BLOCK=(1024,1,1)

    start_time = time.process_time()
    V_np32 = CSR[0]
    E_np32 = CSR[1]
    W_np32 = CSR[2]
    N_np32 = n

    DIST =  [ INF for i in range(0,n*n)]
    VISIT = [ 0 for i in range(0,n*n)]
    PREDIST=[ INF for i in range(0,n*n)]
    # randomSeq = [x for x in range(0,n)]
    # random.shuffle(randomSeq)

    for st in range(0,n):
        DIST[st * n + st] = 0
        VISIT[st * n + st] = 1
    
    DIST_np32 = np.copy(DIST).astype(np.int32)
    VISIT_np32 = np.copy(VISIT).astype(bool)
    PREDIST_np32=np.copy(PREDIST).astype(np.int32)
    # RANDOMSEQ_np32 = np.copy(randomSeq).astype(np.int32)

    KERNEL(
        drv.In(V_np32), drv.In(E_np32), drv.In(W_np32), drv.In(N_np32),
        drv.In(VISIT_np32), drv.InOut(DIST_np32), drv.In(PREDIST_np32),
        block=BLOCK, grid=GRID
    )
    end_time = time.process_time()
    timeCost = end_time - start_time
    result = Result(dist = DIST_np32, timeCost = timeCost, msg = para.msg, graph = para.CSR, graphType = 'CSR')

    if pathRecording:
        result.calcPath()

    return result

    
    