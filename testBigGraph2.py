from utils.myPrint import PRINT_blue
from utils.myPrint import PRINT_red
from utils.check import check

from pretreat import read
from calc import calc

from time import time

if __name__ == "__main__":
    filename = f"./data/data20000000_109000000.txt"

    g = read(filename = filename, detail = True)
    print(g.msg)

    t1 = time()
    res1 = calc(graph = g, useCUDA = False, srclist = 2)
    t2 = time()
    print(f"串行计算完毕, timeCost = {t2 - t1}")

    print(res1.timeCost)
