
from utils.myPrint import PRINT_blue
from utils.myPrint import PRINT_red
from utils.check import checkBool

from calc import calc
from pretreat import read
from time import time

g = read(filename = './data/data_10_20')

print(g.graph)

# CPU 
r1 = calc(graph = g, useCUDA = False, srclist = None)

# GPU 
r2 = calc(graph = g, useCUDA = True, srclist = None)

print(r1.dist)
print(r2.dist)

print(checkBool(r1.dist, r2.dist))
