import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import time
import numpy as np
import random
import os

from classes.result import Result
from utils.settings import INF

kernelName="kernelForAPSP"
fr=open("./method/apsp/cu/spfa.cu","r",encoding='utf-8')
kernelCode=fr.read()
fr.close()
mod=SourceModule(kernelCode)
KERNEL=mod.get_function(kernelName)
BLOCK=(1024,1,1)
GRID=(4096,1,1)

def spfa(CSR,n,pathRecording = False):
    """
	function: use Bellman-Ford algorithm in GPU to solve the APSP . 
	
	parameters:  
		CSR: CSR graph data. (more info please see the developer documentation) .
        n: the number of the vertexs in the graph.
        pathRecordingBool: record the path or not.
	
	return: Result(class).(more info please see the developer documentation) .      
    """

    logger.info("turning to func spfa-gpu-apsp")

    CSR = para.CSR
    n = para.n 
    pathRecording = para.pathRecordingBool

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

    # DIST =  [ INF for i in range(0,n*n)]
    # VISIT = [ 0 for i in range(0,n*n)]
    # PREDIST=[ INF for i in range(0,n*n)]
    # randomSeq = [x for x in range(0,n)]
    # random.shuffle(randomSeq)

    DIST_np32  = np.full( (n*n, ), INF).astype(np.int32)
    PREDIST_np32=np.full( (n*n, ), INF).astype(np.int32)
    # VISIT_np32 = np.full( (n*n, ), 0).astype(bool)
    for st in range(0,n):
        DIST_np32[st * n + st] = np.int32(0)
        # VISIT_np32[st * n + st] = np.bool(True)
    
    # DIST_np32 = np.copy(DIST).astype(np.int32)
    # VISIT_np32 = np.copy(VISIT).astype(bool)
    # PREDIST_np32=np.copy(PREDIST).astype(np.int32)
    # RANDOMSEQ_np32 = np.copy(randomSeq).astype(np.int32)

    KERNEL(
        drv.In(V_np32), drv.In(E_np32), drv.In(W_np32), drv.In(N_np32),
        drv.InOut(DIST_np32), drv.In(PREDIST_np32),
        block=BLOCK, grid=GRID
    )
    # KERNEL(
    #     drv.In(V_np32), drv.In(E_np32), drv.In(W_np32), drv.In(N_np32),
    #     drv.In(VISIT_np32), drv.InOut(DIST_np32), drv.In(PREDIST_np32),
    #     block=BLOCK, grid=GRID
    # )
    end_time = time.process_time()
    timeCost = end_time - start_time
    result = Result(dist = DIST_np32, timeCost = timeCost)
    return result

    
    