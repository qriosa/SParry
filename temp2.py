from calc import calc
import numpy as np
from pretreat import read

filename = "./data/test.txt"
graph = read(filename = filename, # 传入的数据是文件
                 method = "spfa", # 使用的算法是 spfa
                 detail = True, # 记录图中的细节
                 directed = False) # 图为无向图

res = calc(graph = graph, # 传入的图数据类
               useCUDA = True, # 使用 CUDA 并行加速
               srclist = 0) # 源点为 0 号结点
res.drawPath()