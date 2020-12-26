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
        class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter') 
    
    return: 
        class, Result object. (see the 'SPoon/classes/result.py/Result')    
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
        CSR: CSR graph data. (more info please see the developer documentation) .
        n: int, the number of the vertices in the graph.
        s: int, the source vertex.
        pathRecordBool: bool, record the path or not.
        block: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        grid: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.
    
    return: 
        class, Result object. (see the 'SPoon/classes/result.py/Result') 
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

    # 距离数组
    dist = np.full((n,), INF).astype(np.int32)
    dist[s] = 0

    # vis
    vis = np.full((n, ), 1).astype(np.int32)
    vis[s] = np.int32(0)

    # predist
    predist = np.full((n, ), INF).astype(np.int32)

    # 获取函数
    dij_sssp_cuda_fuc = mod.get_function('dijkstra')  

    # 开始跑
    dij_sssp_cuda_fuc(drv.In(V), 
                    drv.In(E), 
                    drv.In(W),
                    drv.In(n),  
                    drv.In(vis), 
                    drv.InOut(dist), 
                    drv.In(predist), 
                    block=BLOCK, grid=GRID)  

    timeCost = time() - t1

    # 结果
    return dist, timeCost


# 就是针对直接 无法放入 GPU 的显存中的情况，这里分两种情况解决测试
# 一是 不使用多流，直接分段进行，使用默认流。
# 二是 使用多流但是需要满足 流数×每次拷贝的边数 不可以超过剩余的显存空间 
# 但是这个也是可以通过 先确定流数再确定边数 还是先确定边数再确定流数s
# 但是 CU 中的函数是一致的 无需更改
def noStream(CSR, n, m, s, part, pathRecordBool, BLOCK, GRID):
    """
    function: 
        use dijkstra algorithm in GPU to solve the SSSP. 
    
    parameters:  
        CSR: CSR graph data. (more info please see the developer documentation) .
        n: int, the number of the vertices in the graph.
        m: int, the number of edges in the graph.
        s: int, the source vertex.
        pathRecordBool: bool, record the path or not.
        block: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        grid: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.
    
    return: 
        class, Result object. (see the 'SPoon/classes/result.py/Result') 
    """

    logger.debug("turning to func dijkstra-gpu-sssp divide")

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

        # temp_gpu = drv.mem_alloc(temp.nbytes)
        # drv.memcpy_htod(temp_gpu, temp)

        bases.append(temp)
        
        Es.append(E[i * part:(i + 1) * part])
        Ws.append(W[i * part:(i + 1) * part])


    # 申请变量空间
    dist = np.full((n * 1, ), INF).astype(np.int32)
    vis = np.full((n * 1, ), 0).astype(np.int32)
    predist = np.full((n * 1, ), INF).astype(np.int32)

    # 多/全源的时候 若直接把 dist 放入太大 则可能只能通过多次单源来解决了
    # 为各个源点初始化

    dist[s] = np.int32(0)
    vis[s] = np.int32((V[s + 1] + part - 1) // part - (V[s]) // part) # 计算其会被分割在几个part中

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

    flag = np.full((1, ), 0).astype(np.int32)
    flag_gpu = drv.mem_alloc(flag.nbytes)

    # 基地址的空间申请
    base_gpu = drv.mem_alloc(bases[0].nbytes)

    for j in range(n):

        flag[0] &= np.int32(0)
        drv.memcpy_htod(flag_gpu, flag)    
        
        for i in range(partNum):
            # 拷贝基地址 bases[i] 到 GPU
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

    # 结果
    return dist, timeCost


# 下面是没有使用的
'''

# V 是全部进去了的 dist vis predist 都是全部进去了
def divide(CSR, n, m, s, streamNum = None, part = None, pathRecordBool = False):
    """
    因为存在当图的大小过于巨大，而无法直接放入显存中，这里就是分段拷贝数据进入显存进行计算的方法。
    此函数，不进行多流的应用直接应用分多次拷贝即可，使用默认流
    注意 这里的 m 必须是明确的 m，即无向图应该在传入之前就算好 2 * m 的有向边 再传入进来
    """
    
    with open(cuFilepath, 'r', encoding = 'utf-8') as f:
        cuf = f.read()
    mod = SourceModule(cuf)

    # 起始时间
    t1 = time()

    V, E, W = CSR[0], CSR[1], CSR[2]

    BLOCK = (1024, 1, 1)
    GRID = (1, 1, 1)

    # 如果没有指定一次拷贝的边的数量，则每次拷贝的值我们来定
    # 暂时是写死的 但是 后面应按照实际运行中不超过最大的值来确定每次拷贝进去的边的数量
    if part == None:
        part = np.int32(1024 * 2000) # 一个流拷贝进去的边的数量
    else:
        part = np.int32(part)

    # 这里的 m 无需再乘 2 因为传入的数据必须针对无向边用两条有向边来表示了
    partNum = (m + part - 1) // part # 计算一共有多少边的块数据需要拷贝

    # 流的数量也是一样的 如果没指定我们就用 8  虽然我也不知道为什么用 8
    # 后续还是应该调研一下 流的数量和 GPU 的 SM　数量的关系　以及一些其他的关系
    # 这里同时还应该保证　流的数量　乘　每次拷贝的边的数量不应该超过当前剩余的显存数量否则就不应该这么多流
    if streamNum == None:
        streamNum = 8     # 流的数量

    # # 判断 是否超出目前还剩的显存
    # if streamNum * palapal:
    #     streamNum = ?
    #     or part = ?

    streams = [] # 将流放进一个列表进行存储
    bases = [] # 本流拷贝的数据 part 是从哪个点开始计算偏移的

    Es = []# 切好的 E, S
    Ws = []

    # 实例化流类
    for i in range(streamNum):
        streams.append(drv.Stream())
        
    # 按照分块构建图的各个部分 同时切分好每个部分的起点 并拷贝到GPU中
    for i in range(partNum):

        # 相当于每个的断开点
        temp = np.full((n, ), i * part).astype(np.int32)

        temp_gpu = drv.mem_alloc(temp.nbytes)
        drv.memcpy_htod(temp_gpu, temp)

        bases.append(temp_gpu)
        
        Es.append(E[i * part:(i + 1) * part])
        Ws.append(W[i * part:(i + 1) * part])

    # 通信确保当前所有流空闲了
    for i in range(streamNum):
        streams[i].synchronize()

    # 申请变量空间
    dist = np.full((n * 1, ), INF).astype(np.int32)
    vis = np.full((n * 1, ), 0).astype(np.int32)
    predist = np.full((n * 1, ), INF).astype(np.int32)

    # 多/全源的时候 若直接把 dist 放入太大 则可能只能通过多次单源来解决了
    # 为各个源点初始化
    # vis 的作用不再是 vis bool 而是一个 int 表示其点的松驰能力
    # 某一个点在某一个 part 中都有一次松驰能力，因此 某个点在几个 part 中其松驰能力就是几 

    dist[s] = np.int32(0)
    vis[s] = np.int32((V[s + 1] + part - 1) // part - (V[s]) // part) # 计算其会被分割在几个part中

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
    divide_cuda_fuc = mod.get_function('divide')

    flag = np.full((1, ), 0).astype(np.int32)
    flag_gpu = drv.mem_alloc(flag.nbytes)

    for j in range(n):

        flag[0] &= np.int32(0)
        drv.memcpy_htod(flag_gpu, flag)    
        
        for i in range(partNum):

            # 这里只是一个临时之法 比较一般情况下是先拷贝的流会先完成 实际上还是应该任务调度的
            freeStream = None
            while freeStream == None:
                for k in range(streamNum):
                    if streams[k].is_done() == True:
                        freeStream = streams[k]
                        break
            
            divide_cuda_fuc(V_gpu, 
                            gpuarray.to_gpu_async(Es[i], stream = freeStream),  
                            gpuarray.to_gpu_async(Ws[i], stream = freeStream), 
                            n_gpu, 
                            flag_gpu, 
                            bases[i], 
                            part_gpu, 
                            vis_gpu, 
                            dist_gpu,
                            predist_gpu, 
                            block = BLOCK, 
                            grid = GRID, 
                            stream = freeStream)

        for i in range(streamNum):
            streams[i].synchronize()

        drv.memcpy_dtoh(flag, flag_gpu)

        if flag[0] == 0:
            break

    drv.memcpy_dtoh(dist, dist_gpu)

    timeCost = time() - t1

    # 结果
    result = Result(dist = dist, timeCost = timeCost)
    
    if pathRecordBool:
        result.calcPath(CSR = CSR)

    return result


# 目前又感觉下面这个其实没啥用 应该可以归属到上一个中间去 
# 流的数量过多以后 就会导致一流一个 part 不足以计算完，因此必须复用流 就和上面一样了 
# 下面这个就是有多流了 但是流的数量×一次拷贝的数量不可以超过剩余的显存数量，否则会溢出
def useStream(CSR, n, m, s, part = None, pathRecordBool = False):
    """
    这个就是使用多流的方案的函数 
    流的数量通过边的数量和 part 值来确定 但是不应该超过显存
    流的数量一定要保证乘上每个流拷贝的数量要在剩余容量中
    否则有可能会溢出的 这个或许添加一个 device 的函数来判断是否满足条件
    """

    with open(cuFilepath, 'r', encoding = 'utf-8') as f:
        cuf = f.read()
    mod = SourceModule(cuf)

    # 起始时间
    t1 = time()

    V, E, W = CSR[0], CSR[1], CSR[2]

    BLOCK = (1024, 1, 1)
    GRID = (1, 1, 1)

    # 如果没有指定 part 就是我一个人快乐了  个人感觉 part 越大越好
    # 但是仍然需要进一步的实验来确定是否是 part 越大越好 
    if part == None:
        part = np.int32(1024 * 2000) # 一次拷贝的边的数量
    else:
        part = np.int32(part)
    
    # 这里还没有考虑 当流的数量和 part 乘起来超出显存了怎么办？
    # 要么就是依据显存的剩余量看能放入多少个 part 来 确定流的数量 然后这些流进行复用直到把所有边都传入解决了
    streamNum = (m + part - 1) // part  # 由数量确定流数

    # 申请变量空间
    # 这里的vis的表示该点的松驰能力
    dist = np.full((n, ), INF).astype(np.int32)
    vis = np.full((n, ), 0).astype(np.int32)
    predist = np.full((n, ), INF).astype(np.int32)    

    streams = [] # 将流放进一个列表进行存储
    bases = [] # 本流拷贝的数据 part 是从哪个点开始计算偏移的

    Es = [] # 切好的 E, S
    Ws = []

    for i in range(streamNum):

        streams.append(drv.Stream())

        # 相当于每个的断开点
        temp = np.full((n, ), i * part).astype(np.int32)

        temp_gpu = drv.mem_alloc(temp.nbytes)
        drv.memcpy_htod(temp_gpu, temp)

        bases.append(temp_gpu)
        
        Es.append(E[i * part:(i + 1) * part])
        Ws.append(W[i * part:(i + 1) * part])


    for i in range(streamNum):
        streams[i].synchronize()

    # 为各个源点初始化
    dist[s] = np.int32(0)
    vis[s] = np.int32((V[s + 1] + part - 1) // part - (V[s]) // part) # 计算其会被分割在几个part中


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
    useStream_cuda_fuc = mod.get_function('divide')

    # 仅仅 copy part 这么多个点进去，然后 E 和 W 则是这些点能够到达的范围 也就是不一定 EW也只有 part 这么多 一般情况下会多一些
    
    flag = np.full((1, ), 0).astype(np.int32)
    flag_gpu = drv.mem_alloc(flag.nbytes)

    for j in range(n):
        # 这里不需要各个流同步等待一下吗？？？？？？？？？？？？？

        flag[0] &= np.int32(0)
        drv.memcpy_htod(flag_gpu, flag)    
        
        for i in range(streamNum):
            useStream_cuda_fuc(V_gpu, 
                            gpuarray.to_gpu_async(Es[i], stream = streams[i]),  
                            gpuarray.to_gpu_async(Ws[i], stream = streams[i]), 
                            n_gpu, 
                            flag_gpu, 
                            bases[i], 
                            part_gpu, 
                            vis_gpu, 
                            dist_gpu,
                            predist_gpu, 
                            block = BLOCK, 
                            grid = GRID, 
                            stream = streams[i])

        for i in range(streamNum):
            streams[i].synchronize()

        drv.memcpy_dtoh(flag, flag_gpu)

        if flag[0] == 0:
            break

    drv.memcpy_dtoh(dist, dist_gpu)

    timeCost = time() - t1

    # 结果
    result = Result(dist = dist, timeCost = timeCost)
    
    if pathRecordBool:
        result.calcPath(CSR = CSR)

    return result
'''