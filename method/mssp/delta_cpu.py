from time import time
import numpy as np

from method.sssp.delta_cpu import delta_stepping as delta_sssp
from classes.result import Result
from utils.debugger import Logger

logger = Logger(__name__)

def delta_stepping(para):
    """
    function: 
        use delta_stepping algorithm in CPU to solve the MSSP. 
    
    parameters:  
        class, Parameter object. (see the 'sparry/classes/parameter.py/Parameter').
    
    return: 
        class, Result object. (see the 'sparry/classes/result.py/Result').
    """

    logger.debug("turning to func delta_stepping-cpu-mssp")

    t1 = time()

    srclist, pathRecordBool = para.srclist.copy(), para.pathRecordBool

    dist = []

    for s in srclist:
        para.srclist = s
        resulti = delta_sssp(para)
        dist.append(resulti.dist)    
    
    para.srclist = srclist
    dist = np.array(dist)

    timeCost = time() - t1

    # result
    result = Result(dist = dist, timeCost = timeCost, graph = para.graph)

    if pathRecordBool:
        result.calcPath()
            
    return result
