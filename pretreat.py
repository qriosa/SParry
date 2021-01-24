import numpy as np
import os

from classes.graph import Graph
from utils.settings import INF
from utils.debugger import Logger

logger = Logger(__name__)

def read(CSR = None, matrix = None, edgeSet = None, filename = "", method = "dij", detail = False, directed = "Unknown", delta = 3):
    """
    function: 
        convert a graph from [CSR/matrix/edgeSet/file] 
            to Graph object(see the 'SPoon/classes/graph.py/Graph')
            as the paremeter 'graph' of the func calc(see the 'SPoon/calc.py/calc')
    
    parameters: 
        CSR: tuple, optional, a 3-tuple of integers as (V, E, W) about the CSR of graph data.
            (more info please see the developer documentation).
        matrix: matrix/list, optional, the adjacency matrix of the graph.
            (more info please see the developer documentation).
        edgeSet: tuple, optional, a 3-tuple of integers as (src(list), des(list), val(list)) about the edge set.
            (more info please see the developer documentation).
        filename: str, optional, the name of the graph file, optional, and the graph should be list as edgeSet.
            (more info please see the developer documentation).
        method: str, optional, the shortest path algorithm that you want to use, only can be [dij, spfa, delta, fw, edge].
            (more info please see the developer documentation).
        detail: bool, optional, default 'False', whether you need to read the detailed information of this picture or not.
            If you set True, it will cost more time to get the detail information.
        directed: bool, optional, default 'False', the graph is drected or not.
            It will be valid, only the 'filename' parameter is set.
        delta: int, optional, default 3, the delta of the delta-stepping algorithom.
            It will be valid, only you choose method is delta-stepping.

        ATTENTION:
            CSR, matrix, edgeSet and filename cann't be all None. 
            You must give at least one graph data.
            And the priority of the four parameters is:
                CSR > matrix > edgeSet > filename.

    return:
        class, Graph object. (see the 'SPoon/classes/graph.py/Graph')     
    """
    
    """
    暂时是用于读图 默认读图是转化为 CSR  detail 是否需要求图中的各种细节
    delta 是默认的 3
    优先级 CSR -> matrix -> edgeSet -> filename
    """

    logger.info("entering pretreat.read func.")

    # 不能全为空
    if(type(CSR) == type(None) and type(matrix) == type(None) and type(edgeSet) == type(None) and filename == ""):
        raise Exception("CSR, matrix, edgeSet and filename can not be all NULL!") 

    CSR = np.array(CSR)
    edgeSet = np.array(edgeSet)
    matrix = np.array(matrix)

    # 实例化一个图
    g = Graph()
    # 是否有向图
    g.directed = directed
    # delta 的 delta 值
    g.delta = delta

    # 修改本图数据将会采用的方法
    if(method == "dij" or method == "spfa" or method == "delta" or method == "edge"):
        g.method = method
    else:
        raise Exception("method can only be one of ['dij', 'spfa', 'delta', 'edge'], not " + method)

    # 进行类型判断
    # CSR 被输入了
    if((CSR.shape == ()) == False):
        if(type(CSR) != np.ndarray):
            raise Exception("CSR can only be list or numpy.ndarray")

        # 需要将 CSR 修改为 edgeSet
        if g.method == "edge":

            logger.info("processing CSR to edgeSet")

            V = CSR[0]
            E = CSR[1]
            W = CSR[2]
            
            # 赋值给 g
            g.n = len(V)-1
            g.m = len(E)

            # 记录各个点的度
            g.degree = np.zeros(g.n, dtype = np.int32)

            src = []
            des = []
            val = []
            edgeSet=[]
            MAXW = -1

            for u in range(g.n):
                # 修改度
                d = V[u + 1] - V[u]
                g.degree[u] = d

                # 最大度
                if d > g.MAXD:
                    g.MAXD = d
                    g.MAXU = u
                # 最小度
                if d < g.MIND:
                    g.MIND = d
                    g.MINU = u

                for ind in range(V[u],V[u + 1]):
                    
                    src.append(u)
                    des.append(E[ind])
                    val.append(W[ind])

                    # 最大边权
                    g.MAXW = max(g.MAXW, W[ind])
                    # 最小边权
                    g.MINW = min(g.MINW, W[ind])
            
            # edgeSet
            g.graph = [src, des, val]
            # 提示信息
            g.setmsg()

        # 保持 CSR 的数据格式
        else:
            logger.info("processing CSR")

            g.n = len(CSR[0]) - 1 # 从上次的代码中看到的需要-1的 毕竟默认的
            g.m = len(CSR[1]) # 从上次的代码中看到的是不需要加减的
            g.graph = CSR

            # 求图中的各种细节
            if detail:
                # 记录各个点的度
                g.degree = np.zeros(g.n, dtype = np.int32)
                # 平均度
                g.avgDegree = g.m / g.n 

                for i in range(g.n):
                    # 记录当前点的度
                    d = CSR[0][i+1] - CSR[0][i]
                    g.degree[i] = d

                    # 最大度
                    if d > g.MAXD:
                        g.MAXD = d
                        g.MAXU = i
                    # 最小度
                    if d < g.MIND:
                        g.MIND = d
                        g.MINU = i
                    
                    for j in range(CSR[0][i], CSR[0][i+1]):
                        # 最大边权
                        g.MAXW = max(g.MAXW, CSR[2][j])
                        # 最小边权
                        g.MINW = min(g.MINW, CSR[2][j])
                
                # 提示信息
                g.setmsg()

    # edgeSet 被输入了
    elif((edgeSet.shape == ()) == False):
        if(type(edgeSet) != np.ndarray):
            raise Exception("edgeSet can only be list or numpy.ndarray")
        # 输入的数据都不一样长 显示是非法的
        if not (len(edgeSet[0]) == len(edgeSet[1]) == len(edgeSet[2])):
            raise Exception("invalid edgeSet data")
        
        # 不需要切换格式
        if g.method == "edge":
            logger.info("processing edgeSet")

            # 从 0 开始编号 所以点的个数就是需要最大的点编号+1
            g.n = max(max(edgeSet[0]), max(edgeSet[1])) + 1
            g.m = len(edgeSet[0])
            g.graph = edgeSet

            # 求图中的各种细节
            if detail:
                # 平均度
                g.avgDegree = g.m / g.n 

                # 各个点的度
                g.degree = np.zeros(g.n, dtype = np.int32)

                src = g.graph[0]
                des = g.graph[1]
                w = g.graph[2]

                for i in range(g.m):
                    # 记录最大边权
                    g.MAXW = max(g.MAXW, w[i])
                    #记录最小边权
                    g.MINW = min(g.MINW, w[i])

                    # 当前点的度加一
                    g.degree[src[i]] += 1 
                
                # 更新最大度
                g.MAXD = g.degree.max()
                # 更新最大度的结点
                g.MAXU = g.degree.argmax()
                # 更新最小度
                g.MIND = g.degree.min()
                # 更新最小度的结点
                g.MINU = g.degree.argmin()

                # 提示信息
                g.setmsg()

        # 需要切换为 CSR 数据格式
        else:
            logger.info("processing edgeSet to CSR")

            src = edgeSet[0]
            des = edgeSet[1]
            w = edgeSet[2]

            # 设置结点数量和边的数量
            g.n = max(max(src), max(des)) + 1
            g.m = len(src)

            # 构建 V E W
            V = np.zeros(g.n + 1, dtype = np.int32) # 从 0 开始编号，因此需要加一个 1 制造一个多余的点
            E = np.full((g.m,), -1).astype(np.int32)
            W = np.full((g.m,), INF).astype(np.int32)

            # 每个点的度
            g.degree = np.zeros(g.n, dtype = np.int32)

            for i in range(g.m):
                # 统计各个点的度
                g.degree[src[i]] += 1

            for u in range(g.n):
                # 先设置 V 数组
                V[u + 1] = V[u] + g.degree[u]
                
                # 最大度和最大度的点
                if g.MAXD < g.degree[u]:
                    g.MAXD = g.degree[u]
                    g.MAXU = u
                # 最小度和最小度的点
                if g.MIND > g.degree[u]:
                    g.MIND = g.degree[u]
                    g.MINU = u
            
            # 记录每个结点已经放置的边数
            edgeOfV = np.zeros(g.n, dtype = np.int32)
            for i in range(g.m):
                # 再设置 E 和 W
                u = src[i]

                E[V[u] + edgeOfV[u]] = des[i]
                W[V[u] + edgeOfV[u]] = w[i]

                # 当前起点的已经添加了一条边了
                edgeOfV[u] += 1

                # 最大边权
                g.MAXW = max(g.MAXW, w[i])
                # 最小边权
                g.MINW = min(g.MINW, w[i])

            # 设置 graph
            g.graph = [V, E, W]

            # 提示信息
            g.setmsg()        

    # 邻接矩阵
    elif((matrix.shape == ()) == False):
        if(matrix.shape[0] != matrix.shape[1]):
            raise Exception("Matrix must be matrix, not a wrong shape " + str(matrix.shape))
        
        # 结点数量
        g.n = matrix.shape[0]
        g.m = 0

        # 各个点的度
        g.degree = np.zeros(g.n, dtype = np.int32)

        # 转化为 edgeSet
        if(g.method == "edge"):

            logger.info("processing matrix to edgeSet")
            
            src = []
            des = []
            w = []

            for i in range(g.n):
                for j in range(g.n):
                    if(matrix[i][j] < INF):
                        # i -> j
                        src.append(i)
                        des.append(j)
                        w.append(matrix[i][j])

                        # 边的数量和度
                        g.m += 1
                        g.degree[i] += 1

                        # 最大边权 
                        g.MAXW = max(g.MAXW, w[-1])
                        # 最小边权
                        g.MINW = min(g.MINW, w[-1])
                
                # 最大度和最大度的点
                if g.MAXD < g.degree[i]:
                    g.MAXD = g.degree[i]
                    g.MAXU = i
                # 最小度和最小度的点
                if g.MIND > g.degree[i]:
                    g.MIND = g.degree[i]
                    g.MINU = i

            # graph
            g.graph = [src, des, w]

            # 提示信息
            g.setmsg()

        # 转化为 CSR
        else:
            logger.info("processing matrix to CSR")

            # 构建 V E W
            V = np.zeros(g.n + 1, dtype = np.int32) # 从 0 开始编号，因此需要加一个 1 制造一个多余的点
            E = []
            W = []

            for i in range(g.n):
                for j in range(g.n):
                    if matrix[i][j] < INF:
                        # 边的数量和度
                        g.degree[i] += 1
                        g.m += 1

                        # 构建E和W
                        E.append(j)
                        W.append(matrix[i][j])

                        # 最大边权 
                        g.MAXW = max(g.MAXW, W[-1])
                        # 最小边权
                        g.MINW = min(g.MINW, W[-1])

                # 构建 V 当前点的起点加上度就是下个点的
                V[i + 1] = V[i] + g.degree[i] 

                # 最大度和最大度的点
                if g.MAXD < g.degree[i]:
                    g.MAXD = g.degree[i]
                    g.MAXU = i
                # 最小度和最小度的点
                if g.MIND > g.degree[i]:
                    g.MIND = g.degree[i]
                    g.MINU = i
            
            # graph
            g.graph = [V, E, W]

            # 提示信息
            g.setmsg()

    # filename 被输入了
    elif(filename != ""):
        if(os.path.exists(filename) == False):
            raise Exception("no such a file or dictionary named " + filename)
        # 只有输入不是 True 就是无向图 
        # 这样写可以将我们判断的是无/有向图的显示为无/有向图
        # 我们没有判断的就显示为 Unknown 
        if(g.directed != True):
            g.directed = False

        # 尝试打开文件
        try:
            with open(filename, 'r') as f:
                content = f.read()    
        except:
            raise Exception("fail to open file " + filename)

        # 切成列表 同时去掉末尾可能的空行
        contenList = content.split("\n")
        contenList.remove("")

        # 第一行 结点数量和(无)边的数量
        nm_str = contenList[0].split(" ")
        g.n = np.int32(nm_str[0]) # n 不需要额外+1只需要在V里面虚拟结点就好了
        g.m = np.int32(nm_str[1])
        # 是有向
        if g.directed == False:
            g.m *= 2
        # 所有点的度
        g.degree = np.zeros(g.n, dtype=np.int32)
        # 后面的其余的边
        lines = contenList[1:]
        # 暂时存储每条边的起点、终点、边权 
        src = []
        des = []
        w = []

        # 有向边 只有指定有向图才是 否则就是无向图
        if g.directed:
            for line in lines:
                edge = line.split(" ")
                temp1 = np.int32(edge[0])
                temp2 = np.int32(edge[1])
                temp3 = np.int32(edge[2])

                src.append(temp1)
                des.append(temp2)
                w.append(temp3)

                g.degree[temp1] += 1

                # 最大边权
                g.MINW = min(g.MINW, temp3)
                g.MAXW = max(g.MAXW, temp3)
        else:
            for line in lines:
                edge = line.split(" ")
                temp1 = np.int32(edge[0])
                temp2 = np.int32(edge[1])
                temp3 = np.int32(edge[2])

                src.append(temp1)
                des.append(temp2)
                w.append(temp3)

                src.append(temp2)
                des.append(temp1)
                w.append(temp3)

                g.degree[temp1] += 1
                g.degree[temp2] += 1

                # 最大边权
                g.MINW = min(g.MINW, temp3)
                g.MAXW = max(g.MAXW, temp3)

        # 读作 edgeSet
        if g.method == "edge":
            logger.info("processing filelist to edgeSet")

            if detail:
                # 更新最大度
                g.MAXD = g.degree.max()
                # 更新最大度的结点
                g.MAXU = g.degree.argmax()
                # 更新最小度
                g.MIND = g.degree.min()
                # 更新最小度的结点
                g.MINU = g.degree.argmin()

                # 提示信息
                g.setmsg()
            
            g.graph = [src, des, w]

        # 读做 CSR       
        else:
            logger.info("processing filelist to CSR")

            # 构建 V E W
            V = np.zeros(g.n + 1, dtype=np.int32) # 从 0 开始编号，因此需要加一个 1 制造一个多余的点
            E = np.full((g.m,), -1).astype(np.int32)
            W = np.full((g.m,), INF).astype(np.int32)
            # 添加多余的点
            V[g.n] = g.m
        
            for u in range(g.n):
                # 先设置 V 数组
                V[u + 1] = V[u] + g.degree[u]

                # 最大度和最大度的点
                if g.MAXD < g.degree[u]:
                    g.MAXD = g.degree[u]
                    g.MAXU = u
                # 最小度和最小度的点
                if g.MIND > g.degree[u]:
                    g.MIND = g.degree[u]
                    g.MINU = u       
            
            # 记录每个结点已经放置的边数
            edgeOfV = np.zeros(g.n, dtype = np.int32)

            for i in range(g.m):
                # 再设置 E 和 W
                u = src[i]

                E[V[u] + edgeOfV[u]] = des[i]
                W[V[u] + edgeOfV[u]] = w[i]

                # 当前起点的已经添加了一条边了
                edgeOfV[u] += 1

            # 设置 graph
            g.graph = [V, E, W]

            # 提示信息
            g.setmsg()

    else:
        raise Exception("What happend??")

    # 设置数据格式
    g.reshape()

    # 返回图类
    return g

if __name__ == "__main__":
    """
    CSR = None, 
    matrix = None, 
    edgeSet = None, 
    filename = "", 
    method = "dij",
    detail = False, 
    directed = "Unknown", 
    delta = 3
    """
    filename = "data10_20.txt"

    CSR = np.array([np.array([ 0,  3,  5,  9, 15, 19, 24, 27, 32, 36, 40]), np.array([1, 2, 5, 0, 4, 0, 3, 7, 4, 2, 4, 6, 7, 5, 7, 1, 2, 3, 9, 0, 3, 8,
       9, 9, 3, 8, 8, 2, 3, 3, 7, 7, 5, 6, 6, 9, 4, 5, 5, 8]), np.array([9, 5, 8, 9, 8, 5, 2, 9, 3, 2, 2, 6, 6, 2, 6, 8, 3, 2, 2, 8, 2, 4,
       4, 8, 6, 1, 5, 9, 6, 6, 4, 4, 4, 1, 5, 7, 2, 4, 8, 7])])
    
    edgeSet = [[0, 1, 0, 2, 0, 5, 1, 4, 2, 3, 2, 7, 2, 4, 3, 4, 3, 6, 3, 7, 3, 5, 3, 7, 4, 9, 5, 8, 5, 9, 5, 9, 6, 8, 6, 8, 7, 7, 8, 9], [1, 0, 2, 0, 5, 0, 4, 1, 3, 2, 7, 2, 4, 2, 4, 3, 6, 3, 7, 3, 5, 3, 7, 3, 9, 4, 8, 5, 9, 5, 9, 5, 8, 6, 8, 6, 7, 7, 9, 8], [9, 9, 5, 5, 8, 8, 8, 8, 2, 2, 9, 9, 3, 3, 2, 2, 6, 6, 6, 6, 2, 2, 6, 6, 2, 2, 4, 4, 4, 4, 8, 8, 1, 1, 5, 5, 4, 4, 7, 7]]
    
    matrix = [0,1,2,3],[1,0,2,3],[2,2,0,4],[3,3,4,0]

    g = read(matrix = matrix, detail = True, directed=False, method = "edge")
    print(g.msg)
    # print(g.degree)