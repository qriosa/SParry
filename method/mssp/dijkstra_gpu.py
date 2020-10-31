from time import time
import numpy as np

from classes.result import Result
from utils.settings import INF

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

def dijkstra(para):
    from utils.judgeDivide import judge
    
    if judge(para):
        return divide(para.CSR, para.n, para.m, para.srclist, para.part, para.sNum, para.pathRecordingBool, para.BLOCK, para.GRID)
    else:
        return nodivide(para.CSR, para.n, para.srclist, para.pathRecordingBool, para.BLOCK, para.GRID)

def nodivide(CSR, n, srclist, pathRecordingBool, BLOCK, GRID):
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

    if BLOCK == None:
        BLOCK = (1024, 1, 1)
    
    if GRID == None:
        GRID = (512, 1)

    V, E, W = CSR[0], CSR[1], CSR[2] 
    
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


def divide(CSR, n, m, srclist, part, sPart, pathRecordingBool, BLOCK, GRID):
    """
    function: use dijkstra algorithm in GPU to solve the SSSP, but this func can
        devide the graph if it's too large to put it in GPU memory. 
    
    parameters:  
        CSR: CSR graph data. (more info please see the developer documentation) .
        n: the number of the vertexs in the graph.
        m: the number of the edge in the graph.
        srclist: the source list, can be number.(more info please see the developer documentation).
        part: the number of the edges that will put to GPU at a time.
        sPart: the number of source calc a time.
        pathRecordingBool: record the path or not.
    
    return: Result(class).(more info please see the developer documentation) .
    """
    # nostream
    with open(cuFilepath, 'r', encoding = 'utf-8') as f:
        cuf = f.read()
    mod = SourceModule(cuf)

    # 起始时间
    t1 = time()

    if BLOCK == None:
        BLOCK = (1024, 1, 1)
    
    if GRID == None:
        GRID = (10, 1)

    V, E, W = CSR[0], CSR[1], CSR[2]

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

    # copy to device 
    n_gpu = drv.mem_alloc(n.nbytes)
    drv.memcpy_htod(n_gpu, n)

    part_gpu = drv.mem_alloc(part.nbytes)
    drv.memcpy_htod(part_gpu, part)

    V_gpu = drv.mem_alloc(V.nbytes)
    drv.memcpy_htod(V_gpu, V)

    # 获取kernal函数
    noStream_cuda_fuc = mod.get_function('divide')

    sNum = len(srclist) # 源的个数
    # 一次解决多少个单源问题
    if GRID[0] * GRID[1] < sPart:
        GRID = tuple((sPart, 1))
    
    sPartNum = (sNum + sPart - 1) // sPart # 计算这么多个源点 在一次拷贝 sNum 个源点的情况下需要多少次循环

    dist = []

    for i in range(sPartNum):

        # 申请变量空间
        disti = np.full((n * sPart, ), INF).astype(np.int32)
        vis = np.full((n * sPart, ), 0).astype(np.int32)
        predist = np.full((n * sPart, ), INF).astype(np.int32)

        # 为各个源点初始化
        for j in range(sPart):
            # srclist[j]为源点的情况下 
            # 此时 第 j 行就是第j个源点 同时是 这次解决的 sPart 个源点问题中的第 j 个 而不是原序 
            if i * sPart + j >= sNum:
                sNum_kernal = np.int32(j) # 这次实际进入kernal的s个数，处理srclist 尾部不足无法整除
                break

            disti[j * n + srclist[j + i * sPart]] = np.int32(0)
            vis[i * n + srclist[j + i * sPart]] = np.int32((V[srclist[j + i * sPart] + 1] + part - 1) // part - (V[srclist[j + i * sPart]]) // part)


        # copy to device
        dist_gpu = drv.mem_alloc(disti.nbytes)
        drv.memcpy_htod(dist_gpu, disti)

        predist_gpu = drv.mem_alloc(predist.nbytes)
        drv.memcpy_htod(predist_gpu, predist)

        vis_gpu = drv.mem_alloc(vis.nbytes)
        drv.memcpy_htod(vis_gpu, vis)

        sNum_kernal_gpu = drv.mem_alloc(sNum_kernal.nbytes)
        drv.memcpy_htod(sNum_kernal_gpu, sNum_kernal)

        flag = np.full((1, ), 0).astype(np.int32)
        flag_gpu = drv.mem_alloc(flag.nbytes)

        for j in range(n):

            flag[0] &= np.int32(0) # 里面的所有源都没有更新了才可以 break
            drv.memcpy_htod(flag_gpu, flag)    
            
            for k in range(partNum):
                noStream_cuda_fuc(V_gpu, 
                                drv.In(Es[k]),  
                                drv.In(Ws[k]), 
                                n_gpu, 
                                sNum_kernal_gpu,
                                flag_gpu, 
                                bases[k], 
                                part_gpu, 
                                vis_gpu, 
                                dist_gpu,
                                predist_gpu, 
                                block = BLOCK, 
                                grid = GRID)

            drv.memcpy_dtoh(flag, flag_gpu)

            if flag[0] == 0:
                break

        drv.memcpy_dtoh(disti, dist_gpu)
        
        dist.append(disti)

    timeCost = time() - t1

    # 结果
    result = Result(dist = np.array(dist).flatten(), timeCost = timeCost)
    
    if pathRecordingBool:
        result.calcPath(CSR = CSR)

    return result