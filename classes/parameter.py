class Parameter(object):
    """
    function: 
        transfer the parameters in the functions.

    parameters: 
        None, but 'self'.

    attributes:
        graph: class Graph, must, the graph data that you want to get the shortest path.
            (more info please see the developer documentation).
        BLOCK: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        GRID: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.
        useCUDA: bool, use CUDA to speedup or not.
        useMultiPro, bool, use multiprocessing in CPU or not. only support dijkstra APSP and MSSP.
        device: class, Device object. (see the 'SPoon/classes/device.py/Device') 
        srclist: list/int, the source of shortest path problem.
        sourceType: str, the type of the source.
        pathRecordBool: bool, record the path or not.
        part: int, the number of the edges that will put to GPU at a time.(divide algorithm)
    
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

        self.useCUDA = True # 是否使用 CUDA
        self.useMultiPro = False # 是否使用 CPU 多线程

        self.graph = None # Graph 类
        self.srclist = None # 源点的集合 单个源点的[数字编号]、全源的[无]、多源的[list] 
        self.sourceType = None # SSSP APSP MSSP
        self.pathRecordBool = False # 是否记录路径

        # 以下是分块的参数
        self.part = None # 分块中一次拷贝的边的数目
        