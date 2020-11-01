# 该模块是为了判断是否要进行大图分割跳转
# 同时如果需要则进行相应的参数的计算
from utils.debugger import Logger
from classes.device import Device
from utils.settings import forceDivide

from math import sqrt
import numpy as np


def judge_sssp(para):
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
    

    device = para.device
    freeGpuMem = device.free

    # 判断是否满足
    lowLimit = lambda PART, N, FREE: 2*PART + (2*N)/(2*PART) <= 3.6*FREE - 3*N - 4
    getPart = lambda FREE, N: ((3.6*FREE - 3*N - 4) + sqrt((3.6*FREE - 3*N - 4) ** 2 - 4 * (2*N))) / 2 / 2 * 10 // 10

    assert lowLimit(0.5 * sqrt(2*para.n), para.n, device.free), "该算法目前无法在此设备上解决此规模的问题"

    # # 整个 m 条边可以放进去
    if lowLimit(para.m, para.n, device.free) and forceDivide == False:
        return False # 不用分图跳转

    # 如果没有指定一次拷贝的边的数量，则每次拷贝的值我们来定
    # 暂时是写死的 但是 后面应按照实际运行中不超过最大的值来确定每次拷贝进去的边的数量
    # 如果给定了 part 就要计算其给的 part 是否符合要求
    if para.part != None:
        # 超过边总数是没有意义的
        para.part = np.int32(min(para.m, para.part))
        if lowLimit(0.5 * sqrt(2*para.n), para.n, device.free) == False:
            para.part = getPart(device.free, para.n)
            logger.warnning(f"你给的 part 不行 我已经给你修正为一个可以的了 part = {para.part}")
    else:
        # 如果整个能进去就可以直接整个放进去
        if lowLimit(para.m, para.n, device.free):
            para.part = para.m
        else:
            # 获取 part
            para.part = getPart(device.free, para.n) # 一个流拷贝进去的边的数量
    
    # print part
    # print(f'part = {para.part}, m = {para.m}') 
    
    return True # 需要分图跳转


def judge_mssp(para):
    """
    function: 
        determine whether the current graph needs to use graph segmentation.
    
    parameters: 
        a parameters class. (more info please see the developer documentation) .
    
    return: 
        bool.    
    """
    
    freeGpuMem = para.device.free

    if forceDivide == False and (para.n * para.n + para.n + 2 * para.m) <= 0.9 * freeGpuMem:
        return False # 不需要分图 
    else:
        return judge_sssp(para)

def judge_apsp(para):
    """
    function: 
        determine whether the current graph needs to use graph segmentation.
    
    parameters: 
        a parameters class. (more info please see the developer documentation) .
    
    return: 
        bool.    
    """

    freeGpuMem = para.device.free
    
    if forceDivide == False and (para.n * para.n + para.n + 2 * para.m) <= 0.9 * freeGpuMem:
        return False # 不需要分图 
    else:
        return judge_sssp(para)


def judge(para):
    """
    function: 
        determine whether the current graph needs to use graph segmentation.
    
    parameters: 
        a parameters class. (more info please see the developer documentation) .
    
    return: 
        bool.
    """

    # logger
    logger = Logger(__name__)   

    # 获取设备的显卡信息
    device = Device()
    freeGpuMem = device.free
    logger.info(f"freeGpuMem = {freeGpuMem} Bytes = {freeGpuMem / 1024} KB = {freeGpuMem / 1024 / 1024} MB = {freeGpuMem / 1024 / 1024 / 1024} GB")
    
    para.device = device

    if para.sourceType == "SSSP":
        return judge_sssp(para)

    elif para.sourceType == "MSSP":
        return judge_mssp(para)
    
    elif para.sourceType == "APSP":
        return judge_apsp(para)
        