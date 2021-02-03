__global__ void edge(int* src, int* des, int* w, int* n, int* m, int* srcNum, int* dist){ 
	
	const int e0 = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;  
	const int offset = blockDim.x * blockDim.y * blockDim.z; // the number of thread in a block
	const int blockNum = (const int) gridDim.x * gridDim.y; // the number of block

    int e = -1;
	int sn = -1;
    int s = blockIdx.x; // s is source vertex
    int old = -1;
    
    __shared__ int quickBreak[1]; // the exit flag in the block

    while(s < (*srcNum)){ 
        
        sn = (s * (*n)); // calc the offset
        
        while(1){
            e = e0;
            if(e == 0){
                quickBreak[0] = 0;
            } 
            __syncthreads();
            
            while(e < (*m)){
                
                if(dist[des[e] + sn] > dist[src[e] + sn] + w[e]){
                    atomicMin(&dist[des[e] + sn], dist[src[e] + sn] + w[e]);

                    if(dist[des[e] + sn] = dist[src[e] + sn] + w[e]){
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
