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
        use delta_stepping algorithm in GPU to solve the MSSP. 
    
    parameters:  
        class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter') 
    
    return: 
        class, Result object. (see the 'SPoon/classes/result.py/Result')
    """

    with open('./method/mssp/cu/delta.cu', 'r', encoding = 'utf-8') as f:
        cuf = f.read()
    mod = SourceModule(cuf)

    logger.debug("turning to func delta_stepping-gpu-mssp")

    # 起始时间
    t1 = time()

    CSR, n, srclist, delta, pathRecordBool = para.graph.graph, para.graph.n, para.srclist, para.graph.delta, para.pathRecordBool

    V, E, W = CSR[0], CSR[1], CSR[2]
    
    # 线程开启全局变量 
    if para.BLOCK != None:
        BLOCK = para.BLOCK
    else:
        BLOCK = (1024, 1, 1)
    
    if para.GRID != None:
        GRID = para.GRID
    else:
        GRID = (128, 1)

    # 源点的个数
    srcNum = np.int32(len(srclist))
    srclist = np.copy(srclist).astype(np.int32)

    predist = np.full((n * srcNum, ), INF).astype(np.int32)
    dist = np.full((n * srcNum, ), INF).astype(np.int32)

    # 为各个源点初始化
    for i in range(srcNum):
        # i为源点的情况下 
        dist[i * n + srclist[i]] = np.int32(0)

    # 获取函数
    delta_mssp_cuda_fuc = mod.get_function("delta_stepping")

    # 开始跑
    delta_mssp_cuda_fuc(drv.In(V),
                        drv.In(E),
                        drv.In(W),
                        drv.In(n),
                        drv.In(srcNum),
                        drv.In(srclist),
                        drv.In(delta),
                        drv.InOut(dist),
                        drv.In(predist),
                        block = BLOCK,
                        grid = GRID)

    timeCost = time() - t1
    
    # 结果
    result = Result(dist = dist, timeCost = timeCost, graph = para.graph)

    if pathRecordBool:
        result.calcPath()

    return result
