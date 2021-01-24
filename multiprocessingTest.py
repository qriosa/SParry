from multiprocessing import Process, Value, Array
import time

def f(tid, a):
    a[tid] = tid * 2
    

if __name__ == '__main__':
    num = Value('d', 0.0)
    arr = Array('i', range(50))
    
    numThread = 4
    
    runningThres = []
    squeueTids = list(range(50))
    while len(squeueTids) > 0:
        if len(runningThres) < numThread:
            tid = squeueTids.pop()
            p = Process(target=f, args=(tid, arr))
            p.start()
            runningThres.append(p)
        
        for ii in range(len(runningThres)):
            i = len(runningThres) - ii - 1
            p = runningThres[i]
            if not p.is_alive():
                p.join()
                del(runningThres[i])
                
#        time.sleep(0.01)
    
    while len(runningThres) > 0:     
        for ii in range(len(runningThres)):
            i = len(runningThres) - ii - 1
            p = runningThres[i]
            if not p.is_alive():
                p.join()
                del(runningThres[i])
#        time.sleep(0.01)
                
        
        
#    for i in range(50):
#        p = Process(target=f, args=(i, arr))
#        p.start()
#        procs.append(p)
#    
#    for p in procs:
#        if not p.is_alive():
#            p.join()
            
    
#    time.sleep(1)
#    print(num.value)
    print(arr[:])