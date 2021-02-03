//  edgebased one thread represent a edge
__global__ void edge(int* src, int* des, int* w, int *n, int* m, int* dist){ 
	
	const int e0 = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;  
	const int offset = blockDim.x * blockDim.y * blockDim.z; 
	const int blockNum = (const int) gridDim.x * gridDim.y; 

    int e = -1;
	int sn = -1;
    int s = blockIdx.z *(gridDim.x *  gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    int old = -1;
    
    __shared__ int quickBreak[1]; 
    
    while(s < (*n)){ // source vertex must be valid.
        sn = (s * (*n)); // calc the offset.
        
        while(1){
            e = e0;
            quickBreak[0] = 0;

            __syncthreads();
            
            while(e < (*m)){
                
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
