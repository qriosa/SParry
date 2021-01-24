from utils.myPrint import PRINT_blue
from utils.myPrint import PRINT_red
from utils.check import checkBool

from calc import calc
from pretreat import read

import pandas as pd
from time import time
import numpy as np

# 写入CSV中
Ns = []
Ms = []
Methods = [] # 由于pd不支持非数字，所以用0123代表上面的方法
Checks = [] # 都和 dij 的串行相比较
TimeSingleCPUs = []
TimeMultiCPUs = []

def work(n, m):
    temp = np.load(f'./predata/data_{n}_{m}_CSV.npz')
    V, E, W = temp['arr_0'], temp['arr_1'],temp['arr_2']
    CSR = [V, E, W]
    g = read(CSR = CSR)

    # single-CPU 
    t1 = time()
    r1 = calc(graph = g, useCUDA = False, srclist = None)
    t2 = time()
    TimeSingleCPUs.append((t2 - t1) * 100000 // 10 / 10000) 

    # multi-CPU
    t1 = time()
    r2 = calc(graph = g, useCUDA = False, useMultiPro = True, srclist = None)
    t2 = time()
    TimeMultiCPUs.append((t2 - t1) * 100000 // 10 / 10000) 

    Ns.append(n)
    Ms.append(m)
    Methods.append("dij")
    Checks.append(checkBool(r1.dist, r2.dist))
             

if __name__ == "__main__":
    # 节点数的列表
    ns = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]
    # ns = [100]

    # 度的列表, 有一个度为 1 可以展示稀疏图
    ds = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    # ds = [2]

    # CSV name 每次运行都是一个新的文件名
    filename = f"./testResult/apsp/multi/test_{str(time())[11:]}.csv"

    for n in ns:
        for d in ds:
            work(n, n * d)
            # save
            df = pd.DataFrame({'n':Ns, 'm':Ms, 'method':Methods, 'TimeSingleCPUs':TimeSingleCPUs, 'TimeMultiCPUs':TimeMultiCPUs, 'Checks':Checks})
            df.to_csv(filename)