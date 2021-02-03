from time import time
import numpy as np
from math import sqrt

from utils.settings import INF
from classes.result import Result
from utils.debugger import Logger

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda.autoinit import context

cuFilepath = './method/sssp/cu/dijkstra.cu'
logger = Logger(__name__)

def dijkstra(para):
    """
    function: 
        use dijkstra algorithm in GPU to solve the APSP. 
    
    parameters:  
        class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter').
    
    return: 
        class, Result object. (see the 'SPoon/classes/result.py/Result').
    """

    logger.debug("turning to func dijkstra-gpu-sssp")

    from utils.judgeDivide import judge_sssp

    judge_sssp(para)
    
    if para.part != None:
        dist, timeCost = noStream(para.graph.graph, para.graph.n, para.graph.m, para.srclist, para.part, para.pathRecordBool, para.BLOCK, para.GRID)
    else:
        dist, timeCost = direct(para.graph.graph, para.graph.n, para.graph.m, para.srclist, para.part, para.pathRecordBool, para.BLOCK, para.GRID)

    result = Result(dist = dist, timeCost = timeCost, graph = para.graph)

    if para.pathRecordBool:
        result.calcPath()

    return result

def direct(CSR, n, m, s, part, pathRecordBool, BLOCK, GRID):
    """
    function: 
        use dijkstra algorithm in GPU to solve the SSSP. 
    
    parameters:  
        CSR: CSR graph data. (more info please see the developer documentation).
        n: int, the number of the vertices in the graph.
        s: int, the source vertex.
        pathRecordBool: bool, record the path or not.
        block: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        grid: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.
    
    return: 
        class, Result object. (see the 'SPoon/classes/result.py/Result').
    """

    logger.debug("turning to func dijkstra-gpu-sssp no-divide")

    with open(cuFilepath, 'r', encoding = 'utf-8') as f:
        cuf = f.read()
    mod = SourceModule(cuf)

    t1 = time()

    V, E, W = CSR[0], CSR[1], CSR[2]

    if BLOCK is None:
        BLOCK = (1024, 1, 1)
    
    if GRID is None:
        GRID = (1, 1)

    # dist
    dist = np.full((n,), INF).astype(np.int32)
    dist[s] = 0

    # vis
    vis = np.full((n, ), 1).astype(np.int32)
    vis[s] = np.int32(0)

    # predist
    predist = np.full((n, ), INF).astype(np.int32)

    # get function
    dij_sssp_cuda_fuc = mod.get_function('dijkstra')  

    # run!
    dij_sssp_cuda_fuc(drv.In(V), 
                    drv.In(E), 
                    drv.In(W),
                    drv.In(n),  
                    drv.In(vis), 
                    drv.InOut(dist), 
                    drv.In(predist), 
                    block=BLOCK, grid=GRID)  

    timeCost = time() - t1

    # result
    return dist, timeCost


def noStream(CSR, n, m, s, part, pathRecordBool, BLOCK, GRID):
    """
    function: 
        use dijkstra algorithm in GPU to solve the SSSP. 
    
    parameters:  
        CSR: CSR graph data. (more info please see the developer documentation).
        n: int, the number of the vertices in the graph.
        m: int, the number of edges in the graph.
        s: int, the source vertex.
        pathRecordBool: bool, record the path or not.
        block: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        grid: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.
    
    return: 
        class, Result object. (see the 'SPoon/classes/result.py/Result').
    """

    logger.debug("turning to func dijkstra-gpu-sssp divide")

    with open(cuFilepath, 'r', encoding = 'utf-8') as f:
        cuf = f.read()
    mod = SourceModule(cuf)

    # start
    t1 = time()

    V, E, W = CSR[0], CSR[1], CSR[2]

    if BLOCK == None:
        BLOCK = (1024, 1, 1)
    
    if GRID == None:
        GRID = (1, 1)

    partNum = (m + part - 1) // part # calc the batch num

    bases = [] # the base (of offset) in each batch

    Es = [] # split E, S into batch
    Ws = []
        
    for i in range(partNum):

        # as the break point in each batch
        temp = np.full((n, ), i * part).astype(np.int32)

        bases.append(temp)
        
        Es.append(E[i * part:(i + 1) * part])
        Ws.append(W[i * part:(i + 1) * part])


    # malloc
    dist = np.full((n * 1, ), INF).astype(np.int32)
    vis = np.full((n * 1, ), 0).astype(np.int32)
    predist = np.full((n * 1, ), INF).astype(np.int32)

    dist[s] = np.int32(0)
    vis[s] = np.int32((V[s + 1] + part - 1) // part - (V[s]) // part) # calc this will be divided into how many batches

    # copy to device
    dist_gpu = drv.mem_alloc(dist.nbytes)
    drv.memcpy_htod(dist_gpu, dist)

    predist_gpu = drv.mem_alloc(predist.nbytes)
    drv.memcpy_htod(predist_gpu, predist)

    vis_gpu = drv.mem_alloc(vis.nbytes)
    drv.memcpy_htod(vis_gpu, vis)

    n_gpu = drv.mem_alloc(n.nbytes)
    drv.memcpy_htod(n_gpu, n)

    part_gpu = drv.mem_alloc(part.nbytes)
    drv.memcpy_htod(part_gpu, part)

    V_gpu = drv.mem_alloc(V.nbytes)
    drv.memcpy_htod(V_gpu, V)

    # get function
    noStream_cuda_fuc = mod.get_function('divide')

    flag = np.full((1, ), 0).astype(np.int32)
    flag_gpu = drv.mem_alloc(flag.nbytes)

    # malloc base
    base_gpu = drv.mem_alloc(bases[0].nbytes)

    for j in range(n):

        flag[0] &= np.int32(0)
        drv.memcpy_htod(flag_gpu, flag)    
        
        for i in range(partNum):
            # copy base bases[i] to GPU
            drv.memcpy_htod(base_gpu, bases[i]) 

            noStream_cuda_fuc(V_gpu, 
                            drv.In(Es[i]),  
                            drv.In(Ws[i]), 
                            n_gpu, 
                            flag_gpu, 
                            base_gpu, 
                            part_gpu, 
                            vis_gpu, 
                            dist_gpu,
                            predist_gpu, 
                            block = BLOCK, 
                            grid = GRID)

        drv.memcpy_dtoh(flag, flag_gpu)

        if flag[0] == 0:
            break

    drv.memcpy_dtoh(dist, dist_gpu)

    timeCost = time() - t1

    # result
    return dist, timeCost


