from time import time
import numpy as np

from classes.result import Result
from utils.settings import INF
from utils.debugger import Logger

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

cuFilepath = './method/mssp/cu/dijkstra.cu'
logger = Logger(__name__)

def dijkstra(para):
    """
    function: 
        use dijkstra algorithm in GPU to solve the MSSP. 
    
    parameters:  
        class, Parameter object. (see the 'sparry/classes/parameter.py/Parameter').
    
    return: 
        class, Result object. (see the 'sparry/classes/result.py/Result').
    """

    logger.debug("turning to func dijkstra-gpu-mssp")

    from utils.judgeDivide import judge_mssp
    
    tag = judge_mssp(para)

    # the mssp can solve directly
    if tag == 2:
        dist, timeCost = nodivide(para.graph.graph, para.graph.n, para.srclist, para.pathRecordBool, para.BLOCK, para.GRID)
    
    # the mssp can only be divide to solve
    else:
        dist, timeCost = divide(para.graph.graph, para.graph.n, para.graph.m, para.srclist, para.part, para.pathRecordBool, para.BLOCK, para.GRID, tag)

    result = Result(dist = dist, timeCost = timeCost, graph = para.graph)

    if para.pathRecordBool:
        result.calcPath()

    return result

def nodivide(CSR, n, srclist, pathRecordBool, BLOCK, GRID):
    """
    function: 
        use dijkstra algorithm in GPU to solve the APSP. 
    
    parameters:  
        CSR: CSR graph data. (more info please see the developer documentation).
        n: int, the number of the vertices in the graph.
        srclist: [int/array/None] the source list.
        pathRecordBool: bool, record the path or not.
        block: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        grid: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.
    
    return: 
        class, Result object. (see the 'sparry/classes/result.py/Result').
    """

    logger.debug("turning to func dijkstra-gpu-mssp no-divide")

    with open(cuFilepath, 'r', encoding = 'utf-8') as f:
        cuf = f.read()
    mod = SourceModule(cuf)

    t1 = time()

    if BLOCK == None:
        BLOCK = (1024, 1, 1)
    
    if GRID == None:
        GRID = (512, 1)

    V, E, W = CSR[0], CSR[1], CSR[2] 
    
    # the number of source vertices 
    srcNum = np.int32(len(srclist))
    srclist = np.copy(srclist).astype(np.int32)

    # malloc memory
    dist = np.full((n * srcNum, ), INF).astype(np.int32)
    vis = np.full((n * srcNum, ), 1).astype(np.int32)
    predist = np.full((n * srcNum, ), INF).astype(np.int32)

    # init source vertices
    for i in range(srcNum):
        # i is a source vertex 
        dist[i * n + srclist[i]] = np.int32(0)
        vis[i * n + srclist[i]] = np.int32(0)    

    # get function
    dij_mssp_cuda_fuc = mod.get_function('dijkstra')

    # run!
    dij_mssp_cuda_fuc(drv.In(V),
                        drv.In(E),
                        drv.In(W), 
                        drv.In(n),
                        drv.In(srcNum),
                        drv.In(vis),
                        drv.InOut(dist),
                        drv.In(predist),
                        block = BLOCK,
                        grid = GRID)

    timeCost = time() - t1
    
    # result
    return dist, timeCost

def divide(CSR, n, m, srclist, part, pathRecordBool, BLOCK, GRID, tag):
    """
    function: 
        use dijkstra algorithm in GPU to solve the APSP, but this func can devide the graph if it's too large to put it in GPU memory. 
    
    parameters:  
        CSR: CSR graph data. (more info please see the developer documentation) .
        n: int, the number of the vertices in the graph.
        m: int, the number of the edge in the graph.
        srclist: [int/array/None] the source list.
        part: int, the number of the edges that will put to GPU at a time.
        pathRecordBool: bool, record the path or not.
        block: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        grid: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.
        tag: bool, convert MSSP to SSSP, then the SSSP need to devide or not.
    
    return: 
        class, Result object. (see the 'sparry/classes/result.py/Result'). 
    """

    logger.debug("turning to func dijkstra-gpu-mssp divide")

    # start time
    t1 = time()

    # divide sssp don't need to divide graph
    if tag == 0:
        from method.sssp.dijkstra_gpu import direct as dij
    # divide sssp need divide graph
    else:
        from method.sssp.dijkstra_gpu import noStream as dij

    if GRID != None:
        temp = 1
        for i in GRID:
            temp *= i
        if temp > 1:
            GRID = (1, 1)

    for s in srclist:
        disti = dij(CSR, n, m, np.int32(s), part, pathRecordBool, BLOCK, GRID)
        dist.append(disti)    

    timeCost = time() - t1

    # result
    return dist, timeCost