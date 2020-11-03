from time import time
import numpy as np

from classes.result import Result
from utils.debugger import Logger
from method.sssp.dijkstra_cpu import dijkstra as dij_sssp

logger = Logger(__name__)

def dijkstra(para):
    """
    function: 
        use dijkstra algorithm in CPU to solve the APSP. 
    
    parameters:  
        class, Parameter object.
    
    return: 
        class, Result object. (more info please see the developer documentation) .
    """

    logger.info("turning to func dijkstra-cpu-apsp")

    t1 = time()

    CSR, n, pathRecordingBool = para.CSR, para.n, para.pathRecordingBool

    dist = []

    for s in range(n):
        para.srclist = s
        resulti = dij_sssp(para) 
        dist.append(resulti.dist)    
    
    para.srclist = None
    dist = np.array(dist)

    timeCost = time() - t1

    # 结果
    result = Result(dist = dist, timeCost = timeCost)

    if pathRecordingBool:
        result.calcPath(CSR = CSR)

    return result
