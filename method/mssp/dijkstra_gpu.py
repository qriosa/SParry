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
        class, Parameter object.
    
    return: 
        class, Result object. (more info please see the developer documentation) .    
    """

    logger.info("turning to func dijkstra-gpu-mssp")

    from utils.judgeDivide import judge
    
    if judge(para):
        dist, timeCost = divide(para.CSR, para.n, para.m, para.srclist, para.part, para.pathRecordBool, para.BLOCK, para.GRID)
    else:
        dist, timeCost = nodivide(para.CSR, para.n, para.srclist, para.pathRecordBool, para.BLOCK, para.GRID)

    result = Result(dist = dist, timeCost = timeCost, msg = para.msg, graph = para.CSR, graphType = 'CSR')

    if para.pathRecordBool:
        result.calcPath()

    return result

def nodivide(CSR, n, srclist, pathRecordBool, BLOCK, GRID):
    """
    function: 
        use dijkstra algorithm in GPU to solve the APSP. 
    
    parameters:  
        CSR: CSR graph data. (more info please see the developer documentation) .
        n: the number of the vertices in the graph.
        srclist: the source list.
        pathRecordBool: record the path or not.
        block: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        grid: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.
    
    return: 
        Result(class).(more info please see the developer documentation) .
    """

    logger.info("turning to func dijkstra-gpu-mssp no-divide")

    with open(cuFilepath, 'r', encoding = 'utf-8') as f:
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
    return dist, timeCost

def divide(CSR, n, m, srclist, part, pathRecordBool, BLOCK, GRID):
    """
    function: 
        use dijkstra algorithm in GPU to solve the APSP, but this func can devide the graph if it's too large to put it in GPU memory. 
    
    parameters:  
        CSR: CSR graph data. (more info please see the developer documentation) .
        n: the number of the vertices in the graph.
        m: the number of the edge in the graph.
        srclist: the source list.
        part: the number of the edges that will put to GPU at a time.
        pathRecordBool: record the path or not.
        block: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        grid: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks
    
    return: 
        Result(class).(more info please see the developer documentation) .
    """

    logger.info("turning to func dijkstra-gpu-mssp divide")

    with open(cuFilepath, 'r', encoding = 'utf-8') as f:
        cuf = f.read()
    mod = SourceModule(cuf)

    # 起始时间
    t1 = time()

    V, E, W = CSR[0], CSR[1], CSR[2]

    if BLOCK == None:
        BLOCK = (1024, 1, 1)
    
    if GRID == None:
        GRID = (1, 1)

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


    dist = []

    n_gpu = drv.mem_alloc(n.nbytes)
    drv.memcpy_htod(n_gpu, n)

    part_gpu = drv.mem_alloc(part.nbytes)
    drv.memcpy_htod(part_gpu, part)

    V_gpu = drv.mem_alloc(V.nbytes)
    drv.memcpy_htod(V_gpu, V)

    # 多/全源的时候 若直接把 dist 放入太大 则可能只能通过多次单源来解决了
    # 为各个源点初始化
    sNum = len(srclist)

    for i in range(sNum):
        # 申请变量空间
        disti = np.full((n, ), INF).astype(np.int32)
        vis = np.full((n, ), 0).astype(np.int32)
        predist = np.full((n, ), INF).astype(np.int32)
        
        # i为源点的情况下 
        disti[srclist[i]] = np.int32(0)
        vis[srclist[i]] = np.int32((V[srclist[i] + 1] + part - 1) // part - (V[srclist[i]]) // part)

        # copy to device
        dist_gpu = drv.mem_alloc(disti.nbytes)
        drv.memcpy_htod(dist_gpu, disti)

        predist_gpu = drv.mem_alloc(predist.nbytes)
        drv.memcpy_htod(predist_gpu, predist)

        vis_gpu = drv.mem_alloc(vis.nbytes)
        drv.memcpy_htod(vis_gpu, vis)

        # 获取kernal函数
        noStream_cuda_fuc = mod.get_function('divide')

        flag = np.full((n, ), 0).astype(np.int32)
        flag_gpu = drv.mem_alloc(flag.nbytes)

        for j in range(n):

            # 此时的 flag 是一个 n 维数组 每个表示每个源是否更新完毕
            flag &= np.int32(0)
            drv.memcpy_htod(flag_gpu, flag)    
            
            for ii in range(partNum):
                noStream_cuda_fuc(V_gpu, 
                                drv.In(Es[ii]),  
                                drv.In(Ws[ii]), 
                                n_gpu, 
                                flag_gpu, 
                                bases[ii], 
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

        drv.memcpy_dtoh(disti, dist_gpu)
        dist.append(disti)

    timeCost = time() - t1

    # 结果
    return dist, timeCost