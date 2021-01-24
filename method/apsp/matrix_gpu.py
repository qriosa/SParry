#共享内存
import numpy as np
import random
from time import time

from classes.result import Result
from utils.settings import INF

import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule

def matrix(matrix, n, pathRecordBool = False):
    """
    function: 
        use Floyd-Warshall algorithm in GPU to solve the APSP. 
        (more info please see the developer documentation).
    
    parameters:  
        matrix: adjacency matrix of the graph data. 
            (more info please see the developer documentation) .
        n: the number of the vertices in the graph.
        pathRecordBool: record the path or not.
    
    return:
        class, Result object. (see the 'SPoon/classes/result.py/Result')
    """

    with open('./method/apsp/cu/matrix.cu', 'r', encoding = 'utf-8') as f:
        cuf = f.read()
    mod = SourceModule(cuf)

    t1 = time()

    # 获取函数
    vectorAdd_MinSharedMemory = mod.get_function("vectorAdd_MinSharedMemory")

    # blockSize = 16

    n_old = n
    n = ((n_old-1)//16+1)*16
    
    p = np.full((n, n), INF)
    np.fill_diagonal(p,0)
    p[0:n_old,0:n_old] = matrix
    m = 1
    n = np.int32(n)
    m = np.int32(m)
    p = p.astype(np.int32)
    
    
    nThreads = 16  # block 宽度
    nBlocks = int((n + nThreads - 1) / nThreads)
    while m < n - 1:
        c = np.zeros([n, n])
        c = c.astype(np.int32)
        vectorAdd_MinSharedMemory(drv.In(p), 
                                drv.In(n), 
                                drv.InOut(c), 
                                drv.In(np.int32(INF)),
                                block=(nThreads, nThreads, 1), 
                                grid=(nBlocks, nBlocks))
        p = c
        m = 2 * m
    
    timeCost = time() - t1

    # 结果
    result = Result(dist = p[0:n_old,0:n_old], timeCost = timeCost, msg = para.msg, graph = para.matrix, graphType = 'matrix')

    if pathRecordBool:
        result.calcPath()
    
    return result