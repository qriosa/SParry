class Parameter(object):
    """
    function: 
        transfer the parameters in the functions.

    parameters: 
        None, but 'self'.

    attributes:
        BLOCK: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        GRID: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.
        n: int, the number of the vertices in the graph.
        m: int, the number of the edges in the graph.
        useCUDA: bool, use CUDA to speedup or not.
        useMultiPro, bool, use multiprocessing in CPU or not. only support dijkstra APSP and MSSP.
        device: class, Device object. (see the 'SPoon/classes/device.py/Device') 
        CSR: tuple, a 3-tuple of integers as (V, E, W) about the CSR of graph data. (more info please see the developer documentation).
        matrix: matrix, as (n,n), about adjacency matrix of graph data.
        edgeSet: tuple, a 3-tuple of integers as (src(list), des(list), val(list)) about the edge set.
        graphType: str, type of graph. [edgeSet, matrix, CSR].
        method: str, the algorithm. [dij, delta, spfa, fw, edge]
        srclist: list/int, the source of shortest path problem.
        sourceType: str, the type of the problem. [APSP, SSSP, MSSP]
        pathRecordBool: bool, record the path or not.
        delta: int, the delta of delta-stepping algorithm.
        MAXW: int, the max value of the edges.
        MAXU: int, the vertex has the maxOutDegree.
        maxOutDegree: int, the max out degree of the graph.
        part: int, the number of the edges that will put to GPU at a time.(divide algorithm)
        streamNum: int, the number of streams used.
        msg: str, the info of the graph.
    
    method:
        None, but init.
            
    return 
        class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter') 
    """
    
    def __init__(self):

        # 指定kernel使用的grid block
        self.BLOCK = None
        self.GRID = None

        self.device = None # 设备参数

        # 多进程的一个问题 
        self.namename = None

        self.useCUDA = True # 是否使用 CUDA
        self.useMultiPro = False # 是否使用 CPU 多线程

        self.graph = None # Graph 类
        self.srclist = None # 源点的集合 单个源点的[数字编号]、全源的[无]、多源的[list] 
        self.sourceType = None # SSSP APSP MSSP
        self.pathRecordBool = False # 是否记录路径

        # 以下是分块的参数
        self.part = None # 分块中一次拷贝的边的数目
        