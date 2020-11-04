from utils.debugger import Logger

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

logger = Logger(__name__)

def get_node_pos(node_list, radius=1, step=1, step_num=8, center=(0, 0), dim=2):
    """
    function: 
        get the pos of vertex. https://www.jianshu.com/p/b8c5d01a715e

    parameters:
        node_list: the nodes.
    
    """
    if dim < 2:
        raise ValueError('cannot handle dimensions < 2')
    paddims = max(0, (dim - 2))
    odd_all_num = len(node_list)
    node_pos_list = []
    while odd_all_num > 0:
        cur_lever_num = radius * step_num
        if odd_all_num < cur_lever_num:
            cur_lever_num = odd_all_num
        odd_all_num -= cur_lever_num

        theta = np.linspace(0, 1, cur_lever_num + 1)[:-1] * 2 * np.pi
        theta = theta.astype(np.float32)
        pos = np.column_stack([np.cos(theta) * radius, np.sin(theta) * radius,
                               np.zeros((cur_lever_num, paddims))])
        pos = pos.tolist()
        node_pos_list.extend(pos)
        radius += 1
    all_pos = dict(zip(node_list, node_pos_list))
    return all_pos

def draw(path, n, s, graph, graphType):
    """
    function: 
        use path to draw a pic.

    parameters:
        path: list, must,  about precursors of each vertex in each problem.
        n: int, must, the number of vertices.
        s: int , must, the source vertex.
        graph: str/list/tuple, must, the graph data that you want to get the shortest path.(more info please see the developer documentation).
        graphType: str, must, type of the graph data, only can be [matrix, CSR, edgeSet].(more info please see the developer documentation).
    
    return: 
        None, no return.        
    """

    assert (path is not None and n is not None and graph is not None and graphType is not None), "path, n, graph and graphType can not be None!"

    logger.info(f"entering to func draw, n = {n}, graphType = {graphType}")

    G = nx.DiGraph()
    G.add_nodes_from(np.arange(n))

    if(graphType == 'CSR'):
        V, E, W = graph[0], graph[1], graph[2]
        for i in range(n):
            for j in range(V[i], V[i + 1]):
                G.add_edge(i, E[j], weight=W[j])

    elif(graphType == 'matrix'):
        for i in range(n):
            for j in range(n):
                G.add_edge(i, j, weight=graph[i][j])

    elif(graphType == 'edgeSet'):
        src, des, w = graph[0], graph[1], graph[2]
        m = len(src)
        for i in range(m):
            G.add_edge(src[i], des[i], weight=w[i])

    else:
        raise Exception("unknown graphType, can only be 'matrix', 'CSR' and 'edgeSet'")

    pathi = np.array(path[:n]) 
    values = ['green' for i in range(n)]
    values[s] = 'orange'# 源点 
    
    edge_labels=dict([((u,v,),d['weight']) for u,v,d in G.edges(data=True)])

    red_edges = []
    for i in range(n):
        if pathi[i] != -1:
            red_edges.append((i, pathi[i]))
            # 无向图暂时先这样写
            red_edges.append((pathi[i], i))
    
    edge_colors = ['black' if not edge in red_edges else 'red' for edge in G.edges()]

    # pos=nx.spring_layout(G)
    pos = get_node_pos(G.nodes()) # 按照同心圆分布点

    fig, ax = plt.subplots(figsize=(13, 13))
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
    nx.draw(G,pos, node_color = values, with_labels=True, font_weight='bold', node_size = 800, edge_color=edge_colors,edge_cmap=plt.cm.Reds)
    plt.show()