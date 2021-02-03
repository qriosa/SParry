__global__ void kernelForAPSP(int *V, int *E, int *W, int *n, bool *visit, int *dist, int *predist){
    
    const int blockId  = blockIdx.z *(gridDim.x *  gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    const int threadId = threadIdx.z*(blockDim.x * blockDim.y)+ threadIdx.y* blockDim.x+ threadIdx.x;
    const int blockSize =blockDim.x * blockDim.y * blockDim.z;
    const int gridSize = gridDim.x  * gridDim.y  * gridDim.z;
    __shared__ int QuickExit;
    int u=0, st=0, align=0, old=0;
    st = blockId;
    while(st < (*n))
    {
        align = (st * (*n));
        for(int rnd=0;rnd<(*n);rnd++){
            QuickExit = 0;
            u = threadId;
            while(u < (*n)){
                if(visit[u + align]){
                    visit[u + align]=0;
                    for(int adj = V[u];adj<V[u+1];adj++){
                        old=atomicMin( &predist[align + E[adj]] , dist[align + u] + W[adj]);
                    }
                }
                u+=blockSize;
            }
            __syncthreads();
            u=threadId;
            while(u < (*n)){
                if(predist[align + u] < dist[align + u]){
                    dist[align + u] = predist[align + u];
                    visit[align + u] = 1;
                    QuickExit = 1;
                }
                u+=blockSize;
            }
            __syncthreads();
            if(QuickExit==0){
                break;
            }
        }
        __syncthreads();
        st += gridSize;
    }
}