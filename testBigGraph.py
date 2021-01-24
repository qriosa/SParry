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
    res1 = calc(graph = g, useCUDA = True, srclist = 2)
    t2 = time()
    print(f"并行1计算完毕, timeCost = {t2 - t1}")

    t1 = time()
    res2 = calc(graph = g, useCUDA = True, srclist = 2)
    t2 = time()
    print(f"并行2计算完毕, timeCost = {t2 - t1}")

    t1 = time()
    res3 = calc(graph = g, useCUDA = True, srclist = 2)
    t2 = time()
    print(f"并行3计算完毕, timeCost = {t2 - t1}")

    t1 = time()
    res4 = calc(graph = g, useCUDA = True, srclist = 2)
    t2 = time()
    print(f"并行4计算完毕, timeCost = {t2 - t1}")

    t1 = time()
    res5 = calc(graph = g, useCUDA = True, srclist = 2)
    t2 = time()
    print(f"并行5计算完毕, timeCost = {t2 - t1}")    

    t1 = time()
    res6 = calc(graph = g, useCUDA = True, srclist = 2)
    t2 = time()
    print(f"并行6计算完毕, timeCost = {t2 - t1}")  

    check(res1.dist, res2.dist, "串行", "并行")

    print(res1.timeCost)
    print(res2.timeCost)
    print(res3.timeCost)
    print(res4.timeCost)
    print(res5.timeCost)
    print(res6.timeCost)