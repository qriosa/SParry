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
        use edgeSet in GPU to solve the APSP.  (more info please see the developer documentation) .
    
    parameters:  
        class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter')
    
    return: 
        class, Result object. (see the 'SPoon/classes/result.py/Result') 
    """

    logger.debug("turning to func edge-gpu-apsp")

    with open('./method/apsp/cu/edge.cu', 'r', encoding = 'utf-8') as f:
        cuf = f.read()
    mod = SourceModule(cuf)

    # 开始计时
    t1 = time()

    edgeSet, n, m, pathRecordBool = para.graph.graph, para.graph.n, para.graph.m, para.pathRecordBool
    src, des, w = edgeSet[0], edgeSet[1], edgeSet[2] 

    if para.BLOCK != None:
        BLOCK = para.BLOCK
    else:
        BLOCK = (1024, 1, 1)
    
    if para.GRID != None:
        GRID = para.GRID
    else:
        GRID = (512, 1) 

    # 申请变量空间
    dist = np.full((n * n, ), INF).astype(np.int32)

    # 为各个源点初始化
    for i in range(n):
        # i为源点的情况下 
        dist[i * n + i] = np.int32(0) 

       
    edge_apsp_cuda_fuc = mod.get_function('edge')

    # 开始跑
    edge_apsp_cuda_fuc(drv.In(src),
                        drv.In(des),
                        drv.In(w), 
                        drv.In(n),
                        drv.In(m),
                        drv.InOut(dist),
                        block=BLOCK,grid=GRID)

    timeCost = time() - t1
    
    # 结果
    result = Result(dist = dist, timeCost = timeCost, graph = para.graph)

    if pathRecordBool:
        result.calcPath()

    return result
