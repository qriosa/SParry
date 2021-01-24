// 普通 edge 一个线程代表着一个边
__global__ void edge(int* src, int* des, int* w, int *n, int* m, int* dist){ // 每个线程作为一条边 判断两个端点是否发生了改变 w可以不用每次都传吧
	
	const int e0 = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; // 每个thread有自己的编号 
	const int offset = blockDim.x * blockDim.y * blockDim.z; // 一个 block 里面有多少的thread
	const int blockNum = (const int) gridDim.x * gridDim.y; // block 的数量

    int e = -1;
	int sn = -1;
    int s = blockIdx.z *(gridDim.x *  gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    int old = -1;
    
    __shared__ int quickBreak[1]; // block 内部的退出标识
    
    while(s < (*n)){ // 源点也必须有效才行
        sn = (s * (*n)); // 计算出当前源点的偏移
        
        while(1){
            e = e0;
            quickBreak[0] = 0;

            __syncthreads();
            
            while(e < (*m)){
            
                // if (dist[src[e] + sn] > dist[des[e] + sn] + w[e]){
                //     old = atomicMin(&dist[src[e] + sn], dist[des[e] + sn] + w[e]);

                //     if(dist[src[e] + sn] < old){
                //         quickBreak[0] = 1;
                //     }
                // }
                
                if(dist[des[e] + sn] > dist[src[e] + sn] + w[e]){
                    old = atomicMin(&dist[des[e] + sn], dist[src[e] + sn] + w[e]);

                    if(dist[des[e] + sn] < old){
                        quickBreak[0] = 1;
                    }
                }
                e += offset;
            }
            
            __syncthreads();

            if(quickBreak[0] == 0){
                break;
            }
        }
        s += blockNum;
    }
}
