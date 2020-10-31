import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import time
import numpy as np
import random
import os

from classes.result import Result
from utils.settings import INF

kernelName="kernelForMSSP"
fr=open("./method/mssp/cu/spfa.cu","r",encoding='utf-8')
kernelCode=fr.read()
fr.close()
mod=SourceModule(kernelCode)
KERNEL=mod.get_function(kernelName)
BLOCK=(1024,1,1)
GRID=(4096,1,1)

def spfa(para):
    """
	function: use Bellman-Ford algorithm in GPU to solve the MSSP. 
	
	parameters:  
		CSR: CSR graph data. (more info please see the developer documentation) .
        n: the number of the vertexs in the graph.
        srclist: the source list, can be list.(more info please see the developer documentation).
        pathRecordingBool: record the path or not.
	
	return: Result(class).(more info please see the developer documentation) .        
    """
    CSR = para.CSR
    n = para.n
    srclist = para.srclist
    sn = len(para.srclist)
    pathRecording = para.pathRecordingBool

    start_time = time.process_time()
    V_np32 = CSR[0]
    E_np32 = CSR[1]
    W_np32 = CSR[2]
    N_np32 = n
    S_np32 = srclist
    SN_np32= np.int32(sn)

    DIST =  [ INF for i in range(0,n*sn)]
    VISIT = [ 0 for i in range(0,n*sn)]
    PREDIST=[ INF for i in range(0,n*sn)]

    for ind in range(0,sn):
        DIST[ ind * n + srclist[ind]] = 0
        VISIT[ind * n + srclist[ind]] = 1
    
    DIST_np32 = np.copy(DIST).astype(np.int32)
    VISIT_np32 = np.copy(VISIT).astype(bool)
    PREDIST_np32=np.copy(PREDIST).astype(np.int32)

    KERNEL(
        drv.In(V_np32), drv.In(E_np32), drv.In(W_np32), drv.In(N_np32),
        drv.In(S_np32), drv.In(SN_np32), 
        drv.In(VISIT_np32), drv.InOut(DIST_np32), drv.In(PREDIST_np32),
        block=BLOCK, grid=GRID
    )
    end_time = time.process_time()
    timeCost = end_time - start_time
    result = Result(dist = DIST_np32, timeCost = timeCost)
    return result

    
    