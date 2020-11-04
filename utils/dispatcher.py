from classes.parameter import Parameter 
from utils.transfer import transfer as tf
from utils.getIndex import getIndex
from utils.debugger import Logger

import numpy as np

logger = Logger(__name__)

def dispatch(graph, graphType, method, useCUDA, pathRecordBool, srclist, msg, block, grid):
    """
    function: 
        schedule the program by passing in parameters.
    
    parameters: 
        graph: str/list/tuple, must, the graph data that you want to get the shortest path.(more info please see the developer documentation).
        graphType: str, must, type of the graph data, only can be [matrix, CSR, edgeSet].(more info please see the developer documentation).
        method: str, the shortest path algorithm that you want to use, only can be [dij, spfa, delta, fw, edge].
        useCUDA: bool, use CUDA to speedup or not.
        pathRecordBool: bool, record the path or not.
        srclist: int/lsit/None, the source list, can be [None, list, number].(more info please see the developer documentation).
        msg: the info of the graph.
        block: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        grid: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.
    
    return:
        class, Result object. (more info please see the developer documentation).  
    """

    logger.info(f"begin to dispatch ... ")

    ## 输入变量判断是否合法 ##
    graph = np.array(graph)
    assert graph.shape != (), "图数据不能为空"

    assert (graphType == 'CSR'
            or graphType == 'matrix'
            or graphType == 'edgeSet'
            or graphType == None), "graphType 仅可以是 [CSR, matrix, edgeSet] 之一,None指示默认为CSR"

    assert (method == 'dij' 
            or method == 'spfa' 
            or method == 'delta' 
            or method == 'fw'
            or method == 'edge'
            or method == None), "method 仅可是 [dij, spfa, delta, fw, edge] 之一,默认为dij"


    # 实例化一个 parameter
    para = Parameter()

    # 填入 msg
    para.msg = msg

    # method
    para.method = 'dij' if method is None else method

    # 填入指定的grid和block参数，未指定则为空
    para.GRID = grid
    para.BLOCK = block

    # 依据图的类型将图写入类中
    if graphType == 'CSR' or graphType == None:
        graphType = 'CSR'
        para.CSR = graph
    elif graphType == 'matrix':
        para.matrix = graph
    else:
        para.edgeSet = graph
    
    # 传入的图的类型
    para.graphType = graphType

    # 记录路径否
    para.pathRecordBool = pathRecordBool

    # useCUDA 
    para.useCUDA = useCUDA

    # srclist
    if srclist != None:
        para.srclist = np.array(srclist).astype(np.int32)
    else:
        para.srclist = None

    # 判断源点类型
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

    para.msg += f'''
计算方法\tmethod = {para.method}, 
使用CUDA\tuseCUDA = {para.useCUDA}, 
源点列表\tsrclist = {para.srclist}, 
问题类型\tsourceType = {para.sourceType}, 
记录路径\tpathRecord = {para.pathRecordBool}, 
'''

    # 依据使用的方法来选择图数据的类型
    # 矩阵相乘
    if para.method == 'fw':
        if graphType != 'matrix':
            tf(para, 'matrix')
        
        # 若未进入transfer则需要自动计算一些必要的参数值
        if para.n == None or para.m == None:
            getIndex(para)
        
        # 进入 cuda 的函数
        if useCUDA == True:
            # 打印警告 此方法仅仅提供全源计算 其余的计算没有优势可言 可以利用其他方法
            from method.apsp.matrix_gpu import matrix
            return matrix(para) 
        # 使用CPU跑 
        else:
            # 打印警告 此方法仅仅提供全源计算 其余的计算没有优势可言 可以利用其他方法
            from method.apsp.matrix_cpu import matrix
            return matrix(para)
    
    # 边细粒度
    elif para.method == 'edge':
        if graphType != 'edgeSet':
            tf(para, 'edgeSet')
        
        # 若未进入transfer则需要自动计算一些必要的参数值
        if para.n == None or para.m == None:
            getIndex(para)

        # print(para.n,para.m)
        
        # 进入 cuda 的函数 的 edge
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
                
        # 只是使用 CPU 的 edge
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
        # 转化到 CSR 的格式
        if graphType != 'CSR':
            tf(para, 'CSR')
        
        # 若未进入transfer则需要自动计算一些必要的参数值
        if para.n == None or para.m == None:
            getIndex(para) # 尚存在重复计算
        

        if para.method == 'dij':
            # GPU dijkstra
            if useCUDA == True:
                from utils.judgeDivide import judge
                if False:#judge(para):
                    # 跳转到需要分图
                    # 暂时不导入这个
                    # 才写到这里 目前也只写了dij的sssp的 另外的还需要判断
                    if para.sourceType == 'APSP':
                        from method.apsp.dijkstra_gpu import noStream as dij
                        return dij(para)

                    elif para.sourceType == 'MSSP':
                        from method.mssp.dijkstra_gpu import dijkstra as dij
                        return dij(para)

                    else:
                        from method.sssp.dijkstra_gpu import dijkstra as dij
                        return dij(para)
                else:
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
        
        elif para.method == 'spfa':
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
        elif para.method == 'delta':
            if para.delta == None or para.MAXN == None or para.maxOutDegree == None:
                getIndex(para)

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
            raise Exception(f"unkown method of {para.method}")


