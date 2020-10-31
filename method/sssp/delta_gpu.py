import numpy as np
from time import time

from utils.settings import INF
from classes.result import Result

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

def delta_stepping(para):
    """
	function: use delta_stepping algorithm in GPU to solve the SSSP. 
	
	parameters:  
		CSR: CSR graph data. (more info please see the developer documentation) .
        n: the number of the vertexs in the graph.
        s: the source vertex, can be number.(more info please see the developer documentation).
        delta: the delta of this algorithm.
        pathRecordingBool: record the path or not.
	
	return: Result(class).(more info please see the developer documentation) .  
    """
    
    with open('./method/sssp/cu/delta.cu', 'r', encoding = 'utf-8') as f:
        cuf = f.read()
    mod = SourceModule(cuf)

    # 起始时间
    t1 = time()

    CSR, n, s, delta, pathRecordingBool = para.CSR, para.n, para.srclist, para.delta, para.pathRecordingBool

    # 线程开启全局变量 
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

    # 获取函数
    delta_sssp_cuda_fuc = mod.get_function("delta_stepping")

    # 开始跑 
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
    
    # 结果
    result = Result(dist = dist, timeCost = timeCost)

    if pathRecordingBool:
        result.calcPath(CSR = CSR)

    return result