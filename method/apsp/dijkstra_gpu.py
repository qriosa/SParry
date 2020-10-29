from time import process_time as time
import numpy as np

from classes.result import Result
from utils.settings import INF

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from classes.device import Device
from math import sqrt
cuFilepath = './method/apsp/cu/dijkstra.cu'

def dijkstra(CSR, n, pathRecordingBool = False):
    """
	function: use dijkstra algorithm in GPU to solve the APSP. 
	
	parameters:  
		CSR: CSR graph data. (more info please see the developer documentation) .
        n: the number of the vertexs in the graph.
        pathRecordingBool: record the path or not.
	
	return: Result(class).(more info please see the developer documentation) .
    """

    with open(cuFilepath, 'r', encoding = 'utf-8') as f:
        cuf = f.read()
    mod = SourceModule(cuf)

    t1 = time()

    V, E, W = CSR[0], CSR[1], CSR[2]

    BLOCK = (1024, 1, 1)
    GRID = (512, 1, 1)    

    # 申请变量空间
    dist = np.full((n * n, ), INF).astype(np.int32)
    vis = np.full((n * n, ), 1).astype(np.int32)
    predist = np.full((n * n, ), INF).astype(np.int32)

    # 为各个源点初始化
    for i in range(n):
        # i为源点的情况下 
        dist[i * n + i] = np.int32(0)
        vis[i * n + i] = np.int32(0)    


       
    dij_apsp_cuda_fuc = mod.get_function('dijkstra')

    # 开始跑
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
    
    # 结果
    result = Result(dist = dist, timeCost = timeCost)

    if pathRecordingBool:
        result.calcPath(CSR = CSR)

    return result


def noStream(CSR, n, m, part = None, pathRecordingBool = False):
    """
	function: use dijkstra algorithm in GPU to solve the APSP, but this func can
        devide the graph if it's too large to put it in GPU memory. 
	
	parameters:  
		CSR: CSR graph data. (more info please see the developer documentation) .
        n: the number of the vertexs in the graph.
        m: the number of the edge in the graph.
        part: the number of the edges that will put to GPU at a time.
        pathRecordingBool: record the path or not.
	
	return: Result(class).(more info please see the developer documentation) .
    """
    with open(cuFilepath, 'r', encoding = 'utf-8') as f:
        cuf = f.read()
    mod = SourceModule(cuf)

    # 起始时间
    t1 = time()

    V, E, W = CSR[0], CSR[1], CSR[2]

    BLOCK = (1024, 1, 1)
    GRID = (32, 1, 1)

    # 这里的 m 无需再乘 2 因为传入的数据必须针对无向边用两条有向边来表示了
    partNum = (m + part - 1) // part # 计算一共有多少边的块数据需要拷贝

    bases = [] # 本流拷贝的数据 part 是从哪个点开始计算偏移的

    Es = []# 切好的 E, S
    Ws = []
        
    # 按照分块构建图的各个部分 同时切分好每个部分的起点 并拷贝到GPU中
    for i in range(partNum):

        # 相当于每个的断开点
        temp = np.full((n, ), i * part).astype(np.int32)

        temp_gpu = drv.mem_alloc(temp.nbytes)
        drv.memcpy_htod(temp_gpu, temp)

        bases.append(temp_gpu)
        
        Es.append(E[i * part:(i + 1) * part])
        Ws.append(W[i * part:(i + 1) * part])


    # 申请变量空间
    dist = np.full((n * n, ), INF).astype(np.int32)
    vis = np.full((n * n, ), 0).astype(np.int32)
    predist = np.full((n * n, ), INF).astype(np.int32)

    # 多/全源的时候 若直接把 dist 放入太大 则可能只能通过多次单源来解决了
    # 为各个源点初始化
    for i in range(n):
        # i为源点的情况下 
        dist[i * n + i] = np.int32(0)
        vis[i * n + i] = np.int32((V[i + 1] + part - 1) // part - (V[ + i]) // part)

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

    # 获取kernal函数
    noStream_cuda_fuc = mod.get_function('divide')

    flag = np.full((n, ), 0).astype(np.int32)
    flag_gpu = drv.mem_alloc(flag.nbytes)

    for j in range(n):

        # 此时的 flag 是一个 n 维数组 每个表示每个源是否更新完毕
        flag &= np.int32(0)
        drv.memcpy_htod(flag_gpu, flag)    
        
        for i in range(partNum):
            noStream_cuda_fuc(V_gpu, 
                            drv.In(Es[i]),  
                            drv.In(Ws[i]), 
                            n_gpu, 
                            flag_gpu, 
                            bases[i], 
                            part_gpu, 
                            vis_gpu, 
                            dist_gpu,
                            predist_gpu, 
                            block = BLOCK, 
                            grid = GRID)

        drv.memcpy_dtoh(flag, flag_gpu)

        # 确保所有的源都是松驰完毕了才行 
        if (flag == 0).all():
            break

    drv.memcpy_dtoh(dist, dist_gpu)

    timeCost = time() - t1

    # 结果
    result = Result(dist = dist, timeCost = timeCost)
    
    if pathRecordingBool:
        result.calcPath(CSR = CSR)

    return result
