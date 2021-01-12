from utils.myPrint import PRINT_blue
from utils.myPrint import PRINT_red
from utils.check import check

from pretreat import read
from calc import calc

from time import time

if __name__ == "__main__":
    filename = f"./data/data00.txt"

    g = read(filename = filename, detail = True)
    print(g.msg)

    t1 = time()
    res1 = calc(graph = g, useCUDA = False, useMultiPro = False, srclist = None)
    t2 = time()
    print(f"全部串行计算完毕, timeCost = {t2 - t1}")

    r1 = time()
    res2 = calc(graph = g, useCUDA = False, useMultiPro = True, srclist = None, namename = __name__)
    r2 = time()
    print(f"多进程串行计算完毕, timeCost = {t2 - t1}")

    check(res1.dist, res2.dist, "single", "multipro")

    print(res1.timeCost)
    print(res2.timeCost)