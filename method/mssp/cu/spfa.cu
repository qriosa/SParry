__global__ void kernelForMSSP(int *V, int *E, int *W, int *n, int *src, int *sn, bool *visit, int *dist, int *predist){
    int u=0, stInd=0, st=0, align=0, old=0;
    __shared__ int QuickExit;
    const int blockId  = blockIdx.z *(gridDim.x *  gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    const int threadId = threadIdx.z*(blockDim.x * blockDim.y)+ threadIdx.y* blockDim.x+ threadIdx.x;
    const int blockSize= blockDim.x * blockDim.y * blockDim.z;
    const int gridSize = gridDim.x  * gridDim.y  *  gridDim.z;
    if(blockId >= (*sn)) return ;
    if(threadId >= (*n)) return ;
    stInd = blockId;
    st = src[stInd];
    while(stInd < (*sn))
    {
        align = (stInd * (*n));
        while(1){ /* this while can solve a sssp*/ 
            QuickExit = 0;
            u = threadId;
            while(u < (*n)){
                if(visit[u + align]){
                    visit[u + align]=0;
                    for(int adj = V[u]; adj<V[u+1]; adj++){
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
        stInd += gridSize;
    }
}