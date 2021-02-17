from time import process_time as time
import numpy as np
from math import sqrt

from classes.result import Result
from utils.settings import INF
from utils.debugger import Logger

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

cuFilepath = './method/apsp/cu/dijkstra.cu'
logger = Logger(__name__)

def dijkstra(para):
    """
    function: 
        use dijkstra algorithm in GPU to solve the APSP. 
    
    parameters:  
        class, Parameter object. (see the 'sparry/classes/parameter.py/Parameter').
    
    return: 
        class, Result object. (see the 'sparry/classes/result.py/Result').
    """
    logger.debug("turning to func dijkstra-gpu-apsp")

    from utils.judgeDivide import judge_apsp

    tag = judge_apsp(para)

    # 2 is can run apsp directly. put all to the video memory. 
    if tag == 2:
        dist, timeCost = nodivide(para.graph.graph, para.graph.n, para.pathRecordBool, para.BLOCK, para.GRID)
    
    # otherwise it's necessary to devide it into many sssp.
    else:
        dist, timeCost = divide(para.graph.graph, para.graph.n, para.graph.m, para.part, para.pathRecordBool, para.BLOCK, para.GRID, tag)

    result = Result(dist = dist, timeCost = timeCost, graph = para.graph)

    if para.pathRecordBool:
        result.calcPath()
        
    return result

# 整个图拷贝
def nodivide(CSR, n, pathRecordBool, BLOCK, GRID):
    """
    function: 
        use dijkstra algorithm in GPU to solve the APSP. 
    
    parameters:  
        CSR: CSR graph data. (more info please see the developer documentation).
        n: int, the number of the vertices in the graph.
        pathRecordBool: bool, record the path or not.
        block: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        grid: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.
    
    return: 
        class, Result object. (see the 'sparry/classes/result.py/Result').
    """

    logger.debug("turning to func dijkstra-gpu-apsp no-divide")

    with open(cuFilepath, 'r', encoding = 'utf-8') as f:
        cuf = f.read()
    mod = SourceModule(cuf)

    t1 = time()

    V, E, W = CSR[0], CSR[1], CSR[2]

    if BLOCK == None:
        BLOCK = (1024, 1, 1)
    
    if GRID == None:
        GRID = (512, 1)  

    # malloc the space
    dist = np.full((n * n, ), INF).astype(np.int32)
    vis = np.full((n * n, ), 1).astype(np.int32)
    predist = np.full((n * n, ), INF).astype(np.int32)

    # init the all sources
    for i in range(n):
        # i is the source
        dist[i * n + i] = np.int32(0)
        vis[i * n + i] = np.int32(0)    

    dij_apsp_cuda_fuc = mod.get_function('dijkstra')

    # run!
    dij_apsp_cuda_fuc(drv.In(V),
                        drv.In(E),
                        drv.In(W), 
                        drv.In(n),
                        drv.In(vis),
                        drv.InOut(dist),
                        drv.In(predist),
                        block = BLOCK,
                        grid = GRID)

    timeCost = time() - t1
    
    # result
    return dist, timeCost


# I will not divide the apsp, just call sssp many times to solve it. 

def divide(CSR, n, m, part, pathRecordBool, BLOCK, GRID, tag):
    """
    function: 
        use dijkstra algorithm in GPU to solve the APSP, but this func can devide the graph if it's too large to put it in GPU memory. 
    
    parameters:  
        CSR: CSR graph data. (more info please see the developer documentation) .
        n: int, the number of the vertices in the graph.
        m: int, the number of the edge in the graph.
        part: int, the number of the edges that will put to GPU at a time.
        pathRecordBool: bool, record the path or not.
        block: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        grid: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks
        tag: bool, convert APSP to SSSP, then the SSSP need to devide or not.
    
    return: 
        class, Result object. (see the 'sparry/classes/result.py/Result').
    """

    logger.debug("turning to func dijkstra-gpu-apsp divide")

    # start time
    t1 = time()

    # devide sssp is not necessary to divide.
    if tag == 0:
        from method.sssp.dijkstra_gpu import direct as dij
    # divided sssp need to divide graph.
    else:
        from method.sssp.dijkstra_gpu import noStream as dij
    
    # the distence array.
    dist = []

    # goto sssp, so the other block is no use, so I set it as one block.
    if GRID != None:
        temp = 1
        for i in GRID:
            temp *= i
        if temp > 1:
            GRID = (1, 1)

    for s in range(n):
        disti = dij(CSR, n, m, np.int32(s), part, pathRecordBool, BLOCK, GRID)
        dist.append(disti)

    timeCost = time() - t1

    # result
    return dist, timeCost