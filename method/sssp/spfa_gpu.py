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

BLOCK=(1024,1,1)
GRID=(1,1,1)


kernelName='spfaKernelForSSSP'
fr=open("./method/sssp/cu/spfa.cu","r",encoding='utf-8')
kernelCode=fr.read()
fr.close()
mod=SourceModule(kernelCode)
KERNEL=mod.get_function(kernelName)

def spfa(CSR,n,st,pathRecording = False):
    """
	function: use Bellman-Ford algorithm in GPU to solve the SSSP. 
	
	parameters:  
		CSR: CSR graph data. (more info please see the developer documentation) .
        n: the number of the vertexs in the graph.
        srclist: the source list, can be number.(more info please see the developer documentation).
        pathRecordingBool: record the path or not.
	
	return: Result(class).(more info please see the developer documentation) .     
    """
<<<<<<< Updated upstream
=======

    logger.info("turning to func spfa-gpu-apsp")

    CSR = para.CSR
    n = para.n 
    st = para.srclist
    pathRecording = para.pathRecordingBool

    if(para.GRID is not None):
        GRID = para.GRID
    else:
        GRID=(1,1,1)
    
    if(para.BLOCK is not None):
        BLOCK = para.BLOCK
    else:
        BLOCK=(1024,1,1)
    # print(BLOCK,GRID)
>>>>>>> Stashed changes
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

    # randomSeq = [x for x in range(n)]
    # random.shuffle(randomSeq)
    # RANDOMSEQ_np32 = np.copy(randomSeq).astype(np.int32)

    KERNEL(
        drv.In(RST_V_np32), drv.In(RST_E_np32), drv.In(RST_W_np32),
        drv.In(MAP_N_np32), drv.In(VISIT_np32), drv.InOut(DIST_np32),
        block=BLOCK, grid=GRID
    )
    end_time=time.process_time()
    timeCost = end_time - start_time
    result = Result(dist = DIST_np32, timeCost = timeCost)
    return result

    