'''
生成一个图 四个参数分别是 n, m, l, r 
按照约定 生成的图从0开始编号 默认无向图, 联通图

首行，两个正整数，分别表示结点数量和边的数量（无向图, 即无向边的数目）
接下来m行，每行三个正整数，分别表示一条无向边的起点，终点和边权
最后有一个空行，表示文件结束.
文件格式如下：
“
n m
st1 ed1 w1
st2 ed2 w2
... ... ...
stm edm wm

”
'''

from utils.debugger import Logger

import sys
from time import time

from cyaron import *

def generate(filename = 'data0.txt', n = 1000, m = 30000, l = 1, r = 12):
    """
    function: 
        generate a random graph to file. (more info please see the developer documentation). 

    parameters:
        filename: str, the filename of file to save the graph.
        n: int, the number of the vertices in the graph.
        m: int, the number of the edges in the graph.
        l: int, the min value of a edge.
        r: int, the max value of a edge.
    
    return: 
        None, no return.    
    """

    # logger
    logger = Logger(__name__)

    t1 = time()
    
    graph = Graph.UDAG(n, m, weight_limit = (l, r))

    lines = str(graph).split('\n')

    data = f"{n} {m}\n"

    for line in lines:
        aline = line.split(' ')
        data += f"{int(aline[0]) - 1} {int(aline[1]) - 1} {aline[2]}\n"


    with open(f"./data/{filename}", 'w') as f:
        f.write(data)

    t2 = time()
    logger.info(f"生成图完毕, n = {n}, m = {m}, 边权介于 {l} ~ {r} 间。 \n生成图用时 t = {(t2 - t1) * 1000 // 10 / 100} s\n")


def gen(filename = 'data0.txt', n = 1000, l = 1, r = 30, chain = 0.4, flower = 0.35):
    """
    function: 
        generate a chrysanthemum graph to file. (more info please see the developer documentation). 

    parameters:
        filename: str, the filename of file to save the graph.
        n: int, the number of the vertices in the graph.
        l: int, the min value of a edge.
        r: int, the max value of a edge.
        chain: float, probability of chain.
        flower: float, Probability of chrysanthemum.
    
    return: 
        None, no return.    
    """    

    # logger
    logger = Logger(__name__)

    t1 = time()
    
    graph = Graph.tree(n, chain = 0.4, flower = 0.35, weight_limit = (l, r))

    lines = str(graph).split('\n')

    data = f"{n} {n - 1}\n"

    for line in lines:
        aline = line.split(' ')
        data += f"{int(aline[0]) - 1} {int(aline[1]) - 1} {aline[2]}\n"


    with open(f"./data/{filename}", 'w') as f:
        f.write(data)

    t2 = time()
    logger.info(f"生成图完毕, n = {n}, m = {n - 1}, 边权介于 {l} ~ {r} 间。 \nP(chain) = {chain}, P(flower) = {flower}\n生成图用时 t = {(t2 - t1) * 1000 // 10 / 100} s\n")

if __name__ == '__main__':

    if len(sys.argv) == 3:
        n = int(sys.argv[1])
        m = int(sys.argv[2])
        l = 1
        r = 9

    elif len(sys.argv) == 5:
        n = int(sys.argv[1])
        m = int(sys.argv[2])
        l = int(sys.argv[3])
        r = int(sys.argv[4])

    else:
        n = 1000
        m = 30000
        l = 1
        r = 9

    generate(filename = f'data0.txt', n = n, m = m, l = l, r = r)


    # ns = [10000, 1000000, 21000000, 31000000, 41000000, 101000000, 101000000, 101000000, 101000000, 101000000, 101000000, 101000000, 101000000, 101000000]
    # ms = [8000000, 600000000, 71000000, 101000000, 820000000, 1010000000, 2010000000, 6010000000, 80100000000, 801000000000, 180000000000, 3000000000000, 6000000000000, 60000000000000]

    # for i in range(len(ns)):
    #     generate(filename = f'data{ns[i]}_{ms[i]}.txt', n = ns[i], m = ms[i], l = 2, r = (2**31)//ns[i])
    #     print(f"finished {i}/{len(ns)}")
