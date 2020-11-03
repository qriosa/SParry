from time import time
import numpy as np

from classes.result import Result
from utils.debugger import Logger
from method.sssp.delta_cpu import delta_stepping as delta_sssp

logger = Logger(__name__)

def delta_stepping(para):
    """
    function: 
        use delta_stepping algorithm in CPU to solve the APSP. 
    
    parameters:  
        class, Parameter object.
    
    return: 
        class, Result object. (more info please see the developer documentation) .
    """

    logger.info("turning to func delta_stepping-cpu-apsp")

    t1 = time()
    
    CSR, n, delta, MAXN, pathRecordBool = para.CSR, para.n, para.delta, para.MAXN, para.pathRecordBool

    dist = []

    for s in range(n):
        para.srclist = s
        resulti = delta_sssp(para)
        dist.append(resulti.dist)    
    
    para.srclist = None
    dist = np.array(dist)

    timeCost = time() - t1

    # 结果
    result = Result(dist = dist, timeCost = timeCost, msg = para.msg, graph = para.CSR, graphType = 'CSR')

    if pathRecordBool:
        result.calcPath()

    return result