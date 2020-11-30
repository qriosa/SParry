# 该模块是为了判断是否要进行大图分割跳转
# 同时如果需要则进行相应的参数的计算
'''
返回是 0/1 单源需要/不需要分图

返回 2 全/多源不需要分解为单源
返回 0/1 全/多源需要分解单源 且是否分图
'''
from utils.debugger import Logger
from classes.device import Device
from utils.settings import forceDivide

from math import sqrt
import numpy as np

logger = Logger(__name__)

def getPart(FREE, N, M):
    # 只使用多少倍的空闲空间
    K = 0.8 
    b = 2*N - K/8*FREE
    c = M / 2
    x = (-b + sqrt(b ** 2 - 4 * c)) / 2

    x /= 2 # 本来算出来就是 x 的，但是还是报错于是就强制除以 2 了
    
    return np.int32(x)


def judge_sssp(para):
    """
    function: 
        determine whether the current graph needs to use graph segmentation.
    
    parameters: 
        para: class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter') 
    
    return:
        bool, [0/1/2]. (more info please see the developer documentation).  
    """

    """
    SSSP 情况下的拷贝 
    会拷贝进入的东西有: 
        dist, predist, vis 长度都是 n 个 int32， 
        一个 base 作为每部分的起点 长度是 (n / part = partNum) 个 int 32
        flag、n、part 三个 int32
        一个 W，E， 长度都是 part 这么长的 int32 
        则 显存的开销是
        3*n + (n/part + 1) + 3 + (2 * part) [int32] 即是:
        [3*n + (n/part + 1) + 3 + (2 * part)] / 4 [Byte] (PS: int32 为 4 个字节)
    为了便于计算 则将向上取整直接算作整除 +1，同时为了避免某些未知的显存占用 最多只利用 0.9*free 的显存
    则上述式子化简可以得到合理的part 必须满足式子 2*part + (2*n)/(2*part) ≤ 3.6*free - 3*n - 4 
    根据双钩函数的性质，可以知道 2*part 在 sqrt(2n)处取得最小值，
        若 2*part == sqrt(2*n) 时最小值也无法满足上述式子，则本算法无法计算出这个图的最短路径
        若此时满足了最小值等于，则根据猜测的常识：同一个图在同样的条件下 一次拷贝的边的数量越多计算应该会越快 
            因此问题转化为求解满足上述式子的 最大的 part, 
            不等式直接变等式，转化为一个一元二次方程的求根问题: (2*part)² - (3.6*free - 3*n - 4) * (2*part) + 2*n == 0
            根据单调性 part 显然是往大于 sqrt(2*n) 的方向去取 因此应该取大根 并向下取整
            part = ((3.6*free - 3*n - 4) + sqrt((3.6*free - 3*n - 4)² - 4 * (2*n))) / 2 / 2 * 10 // 10 # 向下取整
    """
    
    # 这个 if 说明是从 APSP 或者 MSSP 中来的
    if para.device == None:
        # 获取设备的显卡信息
        para.device = Device()
        freeGpuMem = para.device.free
        logger.info(f"freeGpuMem = {freeGpuMem} Bytes = {freeGpuMem / 1024} KB = {freeGpuMem / 1024 / 1024} MB = {freeGpuMem / 1024 / 1024 / 1024} GB")
        
    device = para.device
    freeGpuMem = para.device.free

    # 避免溢出
    n = int(para.n)
    m = int(para.m)

    # 判断是否满足
    lowLimit = lambda PART, N, FREE: 2*PART + (4*N) <= 0.225*FREE
    # getPart = lambda FREE, N, M: np.int32(0.05*freeGpuMem - N + 0.5*(sqrt((0.1*freeGpuMem - 2*N)**2 - 2*M)))

    assert 0.1*freeGpuMem - 2*n >= sqrt(m/2) * 2, "该算法目前无法在此设备上解决此规模的问题"

    # # 整个 m 条边可以放进去
    if lowLimit(m, n, device.free) and forceDivide == False:
        return 0

    # 如果没有指定一次拷贝的边的数量，则每次拷贝的值我们来定
    # 暂时是写死的 但是 后面应按照实际运行中不超过最大的值来确定每次拷贝进去的边的数量
    # 如果给定了 part 就要计算其给的 part 是否符合要求
    if para.part != None:
        # 超过边总数是没有意义的
        para.part = np.int32(min(m, para.part))
        if lowLimit(0.5 * sqrt(2*n), n, device.free) == False:
            para.part = getPart(device.free, n, m)
            logger.warnning(f"你给的 part 不行 我已经给你修正为一个可以的了 part = {para.part}")
    else:
        # 如果整个能进去就可以直接整个放进去
        if lowLimit(m, n, device.free):
            para.part = m
            
            return 0
        else:
            # 获取 part
            para.part = getPart(device.free, n, m) # 一个流拷贝进去的边的数量
    
    # print part
    logger.info(f'part = {para.part}, m = {m}') 
    
    # 需要分图跳转
    para.part = np.int32(para.part)

    return 1


def judge_mssp(para):
    """
    function: 
        determine whether the current graph needs to use graph segmentation.
    
    parameters: 
        para: class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter') 
    
    return:
        bool, [0/1/2]. (more info please see the developer documentation).  
    """
    
    # 获取设备的显卡信息
    para.device = Device()
    freeGpuMem = para.device.free
    logger.info(f"freeGpuMem = {freeGpuMem} Bytes = {freeGpuMem / 1024} KB = {freeGpuMem / 1024 / 1024} MB = {freeGpuMem / 1024 / 1024 / 1024} GB")

    # 源点的数量
    sNum = len(para.srclist)
    n = int(para.n)
    m = int(para.m)

    # 不需要分图 
    if forceDivide == False and ((3 * sNum + 1) * n + 2 * m) <= 0.8 * freeGpuMem:
        return 2
    
    return judge_sssp(para)

def judge_apsp(para):
    """
    function: 
        determine whether the current graph needs to use graph segmentation.
    
    parameters: 
        para: class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter') 
    
    return:
        bool, [0/1/2]. (more info please see the developer documentation).    
    """

    # 获取设备的显卡信息
    para.device = Device()
    freeGpuMem = para.device.free

    logger.info(f"freeGpuMem = {freeGpuMem} Bytes = {freeGpuMem / 1024} KB = {freeGpuMem / 1024 / 1024} MB = {freeGpuMem / 1024 / 1024 / 1024} GB")
    
    # 不需要分图
    # 这个乘法居然还有数据溢出的问题 目前解决了
    if forceDivide == False:
        n = int(para.n)
        m = int(para.m)

        # 直接相乘 n ** 2 溢出了 32 位整型了
        temp = (3 * n * n + 2 * n + 2 * m)

        if temp < 0.8 * freeGpuMem:
            return 2
    
    return judge_sssp(para)


# def judge(para):
#     """
#     function: 
#         determine whether the current graph needs to use graph segmentation.
    
#     parameters: 
#         para: class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter')
    
#     return:
#         bool, True/False. (more info please see the developer documentation).
#     """

#     # 获取设备的显卡信息
#     device = Device()
#     freeGpuMem = device.free
#     logger.info(f"freeGpuMem = {freeGpuMem} Bytes = {freeGpuMem / 1024} KB = {freeGpuMem / 1024 / 1024} MB = {freeGpuMem / 1024 / 1024 / 1024} GB")
    
#     para.device = device

#     if para.sourceType == "SSSP":
#         return judge_sssp(para)

#     elif para.sourceType == "MSSP":
#         return judge_mssp(para)
    
#     elif para.sourceType == "APSP":
#         return judge_apsp(para)
        