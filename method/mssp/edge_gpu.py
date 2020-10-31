from time import time
import numpy as np

from classes.result import Result
from utils.settings import INF

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

def edge(para):
    """
	function: use edge free in GPU to solve the MSSP. 
        (more info please see the developer documentation) .
	
	parameters:  
		edgeSet: edgeSet graph data. (more info please see the developer documentation) .
        n: the number of the vertexs in the graph.
        m: the number of the edges in the graph.
        srclist: the source list, can be list.(more info please see the developer documentation).
        pathRecordingBool: record the path or not.
	
	return: Result(class).(more info please see the developer documentation) . 
    """

    with open('./method/mssp/cu/edge.cu', 'r', encoding = 'utf-8') as f:
        cuf = f.read()
    mod = SourceModule(cuf)

    edgeSet, n, m, srclist, pathRecordingBool = para.edgeSet, para.n, para.m, para.srclist, para.pathRecordingBool

    # 将 edgeSet 转化为 三个列表
    src = np.array([item[0] for item in edgeSet], dtype = np.int32)
    des = np.array([item[1] for item in edgeSet], dtype = np.int32)
    w = np.array([item[2] for item in edgeSet], dtype = np.int32)

    t1 = time()
    
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

    # 申请变量空间
    dist = np.full((n * srcNum, ), INF).astype(np.int32)

    # 为各个源点初始化 此时的 i 不再是 i 点 而是第 i 个源点
    for i in range(srcNum):
        # i为源点的情况下  这里需要注意下 
        dist[i * n + srclist[i]] = np.int32(0) 
       
    edge_mssp_cuda_fuc = mod.get_function('edge')

    # 开始跑
    edge_mssp_cuda_fuc(drv.In(src),
                        drv.In(des),
                        drv.In(w), 
                        drv.In(n),
                        drv.In(m),
                        drv.In(srcNum),
                        drv.InOut(dist),
                        block = BLOCK,
                        grid = GRID)

    timeCost = time() - t1
    
    # dist 和 path 都遵循着 在第i个单源问题中的真的值
    # path 则具体是第i个单源问题中的真的路径中的点的编号就不考虑某个点是当前的源点的不会用i来表示这个源点了

    # 结果
    result = Result(dist = dist, timeCost = timeCost)

    if pathRecordingBool:
        result.calcPath(edgeSet = edgeSet)

    return result
