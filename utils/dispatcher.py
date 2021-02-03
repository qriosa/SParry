from classes.parameter import Parameter 
from classes.graph import Graph
from utils.debugger import Logger

import numpy as np

logger = Logger(__name__)

def dispatch(graph, useCUDA, useMultiPro, pathRecordBool, srclist, block, grid):
    """
    function: 
        schedule the program by passing in parameters.
    
    parameters: 
        graph: str/list/tuple, must, the graph data that you want to get the shortest path.
            (more info please see the developer documentation).
        useCUDA: bool, use CUDA to speedup or not.
        useMultiPro, bool, use multiprocessing in CPU or not. only support dijkstra APSP and MSSP.
        pathRecordBool: bool, record the path or not.
        srclist: int/lsit/None, the source list, can be [None, list, number].
            (more info please see the developer documentation).
        block: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        grid: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.
    
    return:
        class, Result object. (see the 'SPoon/classes/result.py/Result').
    """

    logger.info(f"entering to dispatch ... ")

    ## input it's valid or not ##
    assert type(graph) == Graph, """
    parameter graph can only be Graph Object.

    -----------------------------------------------------------   
    try use func read() in pretreat.py to pretreat your graph data.
    such as:
        >>> from pretreat import read
        >>> g = read(CSR = CSR, method = 'dij', [,])
        >>> from calc import calc
        >>> result = calc(graph = graph, [,])

    -----------------------------------------------------------
    Please see the tutorials for more infomation.
"""
    # graph = np.array(graph)
    # assert graph.shape != (), "graph data can not be None."

    # assert (graphType == 'CSR'
    #         or graphType == 'matrix'
    #         or graphType == 'edgeSet'
    #         or graphType == None), "graphType can only be one of [CSR, matrix, edgeSet] , default 'CSR'."

    # assert (method == 'dij' 
    #         or method == 'spfa' 
    #         or method == 'delta' 
    #         or method == 'fw'
    #         or method == 'edge'
    #         or method == None), "method can only be one of [dij, spfa, delta, fw, edge], default 'dij'."

    # instantiation parameter
    para = Parameter()

    # recording graph
    para.graph = graph

    # put grid and block
    para.GRID = grid
    para.BLOCK = block

    # recording path or not
    para.pathRecordBool = pathRecordBool

    # useCUDA 
    para.useCUDA = useCUDA

    # srclist
    if srclist != None:
        para.srclist = np.array(srclist).astype(np.int32)
    else:
        para.srclist = None

    # tell the source vertex type
    if srclist == None:
        para.sourceType = 'APSP'
    elif type(srclist) == list or type(srclist) == np.ndarray:
        if len(srclist) == 1:
            para.sourceType = 'SSSP'
        else:
            para.sourceType = 'MSSP'
    elif type(srclist) == int or type(srclist) == np.int32:
        para.sourceType = 'SSSP'
    else:
        raise Exception("undefined srclist type")

    if useMultiPro and useCUDA == False and (para.graph.method != 'dij' or para.sourceType == 'SSSP'):
        raise Exception("can only support dijkstra algorithm to solve APSP and MSSP.")
    
    para.useMultiPro = useMultiPro

    # choose graph type based on method
    # matrix
    if para.graph.method == 'fw':
        #  cuda 
        if useCUDA == True:
            from method.apsp.matrix_gpu import matrix
            return matrix(para) 
        # CPU
        else:
            from method.apsp.matrix_cpu import matrix
            return matrix(para)
    
    # edgebased
    elif para.graph.method == 'edge':
        # CUDA
        if useCUDA == True:
            if para.sourceType == 'APSP':
                from method.apsp.edge_gpu import edge
                return edge(para)
            elif para.sourceType == 'MSSP':
                from method.mssp.edge_gpu import edge
                return edge(para)
            else:
                from method.sssp.edge_gpu import edge
                return edge(para)

        # CPU
        else:
            if para.sourceType == 'APSP':
                from method.apsp.edge_cpu import edge
                return edge(para)
            elif para.sourceType == 'MSSP':
                from method.mssp.edge_cpu import edge
                return edge(para)
            else:
                from method.sssp.edge_cpu import edge
                return edge(para)

    # dij spfa delta
    else:
        if para.graph.method == 'dij':
            # GPU dijkstra
            if useCUDA == True:
                if para.sourceType == 'APSP':
                    from method.apsp.dijkstra_gpu import dijkstra as dij
                    return dij(para)
                elif para.sourceType == 'MSSP':
                    from method.mssp.dijkstra_gpu import dijkstra as dij
                    return dij(para)
                else:
                    from method.sssp.dijkstra_gpu import dijkstra as dij
                    return dij(para)    
    
            # CPU dijkstra
            else:
                if para.sourceType == 'APSP':
                    from method.apsp.dijkstra_cpu import dijkstra as dij
                    return dij(para)
                elif para.sourceType == 'MSSP':
                    from method.mssp.dijkstra_cpu import dijkstra as dij
                    return dij(para)
                else:
                    from method.sssp.dijkstra_cpu import dijkstra as dij
                    return dij(para)
        
        elif para.graph.method == 'spfa':
            # GPU SPFA
            if useCUDA == True:
                if para.sourceType == 'APSP':
                    from method.apsp.spfa_gpu import spfa as spfa
                    return spfa(para)
                elif para.sourceType == 'MSSP':
                    from method.mssp.spfa_gpu import spfa as spfa
                    return spfa(para)
                elif para.sourceType == "SSSP":
                    from method.sssp.spfa_gpu import spfa as spfa
                    return spfa(para)
                else:
                    raise Exception("can not run calculation by undefined calcType")
            
            # CPU SPFA
            else:
                if para.sourceType == 'APSP':
                    from method.apsp.spfa_cpu import spfa as spfa
                    return spfa(para)
                elif para.sourceType == 'MSSP':
                    from method.mssp.spfa_cpu import spfa as spfa
                    return spfa(para)
                elif para.sourceType == "SSSP":
                    from method.sssp.spfa_cpu import spfa as spfa
                    return spfa(para)
                else:
                    raise Exception("can not run calculation by undefined calcType")
        
        # delta
        elif para.graph.method == 'delta':
            # GPU delta
            if useCUDA == True:
                if para.sourceType == 'APSP':
                    from method.apsp.delta_gpu import delta_stepping as delta
                    return delta(para)
                elif para.sourceType == 'MSSP':
                    from method.mssp.delta_gpu import delta_stepping as delta
                    return delta(para)          
                else:
                    from method.sssp.delta_gpu import delta_stepping as delta
                    return delta(para)

            # CPU delta
            else:
                if para.sourceType == 'APSP':
                    from method.apsp.delta_cpu import delta_stepping as delta
                    return delta(para)
                elif para.sourceType == 'MSSP':
                    from method.mssp.delta_cpu import delta_stepping as delta
                    return delta(para)
                else:
                    from method.sssp.delta_cpu import delta_stepping as delta
                    return delta(para)

        else:
            raise Exception(f"unkown method of {para.graph.method}")
