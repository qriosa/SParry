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
        class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter')
    
    return: 
        class, Result object. (see the 'SPoon/classes/result.py/Result') 
    """

    logger.debug("turning to func delta_stepping-cpu-apsp")

    t1 = time()
    
    n, pathRecordBool = para.graph.n, para.pathRecordBool

    dist = []

    for s in range(n):
        para.srclist = s
        resulti = delta_sssp(para)
        dist.append(resulti.dist)    
    
    para.srclist = None
    dist = np.array(dist)

    timeCost = time() - t1

    # 结果
    result = Result(dist = dist, timeCost = timeCost, graph = para.graph)

    if pathRecordBool:
        result.calcPath()

    return result
    