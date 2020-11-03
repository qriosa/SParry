from time import time
import numpy as np

from classes.result import Result
from utils.settings import INF
from utils.debugger import Logger

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

logger = Logger(__name__)

def edge(para):
    """
    function: 
        use edgeSet in GPU to solve the SSSP.  (more info please see the developer documentation) .
    
    parameters:  
        class, Parameter object.
    
    return: 
        class, Result object. (more info please see the developer documentation) . 
    """

    logger.info("turning to func edge-gpu-sssp")

    with open('./method/sssp/cu/edge.cu', 'r', encoding = 'utf-8') as f:
        cuf = f.read()
    mod = SourceModule(cuf)

    # 起始时间
    t1 = time()

    edgeSet, n, m, s, pathRecordingBool = para.edgeSet, para.n, para.m, para.srclist, para.pathRecordingBool
    src, des, w = para.edgeSet[0], para.edgeSet[1], para.edgeSet[2] 

    # 线程开启全局变量 
    if para.BLOCK != None:
        BLOCK = para.BLOCK
    else:
        BLOCK = (1024, 1, 1)
    
    if para.GRID != None:
        GRID = para.GRID
    else:
        GRID = (1, 1)

    dist = np.full((n, ), INF).astype(np.int32)
    dist[s] = np.int32(0)

    # 退出标识
    flag = np.full((m, ), 1).astype(np.int32)

    # 获取函数
    edge_sssp_cuda_fuc = mod.get_function('edge')  

    # 开始跑
    edge_sssp_cuda_fuc(drv.In(src),
                        drv.In(des),
                        drv.In(w),
                        drv.In(m),
                        drv.InOut(dist),
                        drv.In(flag),
                        block = BLOCK, 
                        grid = GRID)
    
    timeCost = time() - t1

    # 结果
    result = Result(dist = dist, timeCost = timeCost)

    if pathRecordingBool:
        result.calcPath(edgeSet = edgeSet)

    return result