from time import time
import numpy as np

from classes.result import Result
from utils.settings import INF

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

def dijkstra(CSR, n, srclist, pathRecordingBool = False):
    """
	function: use dijkstra algorithm in GPU to solve the MSSP. 
	
	parameters:  
		CSR: CSR graph data. (more info please see the developer documentation) .
        n: the number of the vertexs in the graph.
        srclist: the source list, can be list.(more info please see the developer documentation).
        pathRecordingBool: record the path or not.
	
	return: Result(class).(more info please see the developer documentation) .  
    """

    with open('./method/mssp/cu/dijkstra.cu', 'r', encoding = 'utf-8') as f:
        cuf = f.read()
    mod = SourceModule(cuf)

    t1 = time()

    V, E, W = CSR[0], CSR[1], CSR[2]

    BLOCK = (1024, 1, 1)
    GRID = (32, 1, 1)    
    
    # 源点的个数
    srcNum = np.int32(len(srclist))
    srclist = np.copy(srclist).astype(np.int32)

    # 申请变量空间
    dist = np.full((n * srcNum, ), INF).astype(np.int32)
    vis = np.full((n * srcNum, ), 1).astype(np.int32)
    predist = np.full((n * srcNum, ), INF).astype(np.int32)

    # 为各个源点初始化
    for i in range(srcNum):
        # i为源点的情况下 
        dist[i * n + srclist[i]] = np.int32(0)
        vis[i * n + srclist[i]] = np.int32(0)    

    # 获取函数   
    dij_mssp_cuda_fuc = mod.get_function('dijkstra')

    # 开始跑
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
    
    # 结果
    result = Result(dist = dist, timeCost = timeCost)

    if pathRecordingBool:
        result.calcPath(CSR = CSR)

    return result
