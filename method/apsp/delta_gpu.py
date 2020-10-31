from time import time
import numpy as np

from classes.result import Result
from utils.settings import INF

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

def delta_stepping(para):
    """
    function: use delta_stepping algorithm in GPU to solve the APSP. 
    
    parameters:  
        CSR: CSR graph data. (more info please see the developer documentation) .
        n: the number of the vertexs in the graph.
        delta: the delta of this algorithm.
        pathRecordingBool: record the path or not.
    
    return: Result(class).(more info please see the developer documentation) .
    
    尚不稳定
    """

    with open('./method/apsp/cu/delta.cu', 'r', encoding = 'utf-8') as f:
        cuf = f.read()
    mod = SourceModule(cuf)

    # 起始时间
    t1 = time()

    CSR, n, delta, pathRecordingBool = para.CSR, para.n, para.delta, para.pathRecordingBool

    V, E, W = CSR[0], CSR[1], CSR[2]

    # 线程开启全局变量 
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

    # 为各个源点初始化
    for i in range(n):
        # i为源点的情况下 
        dist[i * n + i] = np.int32(0)

    # 获取函数
    delta_apsp_cuda_fuc = mod.get_function("delta_stepping")

    # 开始跑
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
    
    # 结果
    result = Result(dist = dist, timeCost = timeCost)

    if pathRecordingBool:
        result.calcPath(CSR = CSR)

    return result
