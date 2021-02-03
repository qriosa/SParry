__global__ void spfaKernelForSSSP(int *V, int *E, int *W, int *n, bool *visit,int *dist){
    int old=0, u, v;
    __shared__ int QuickExit;
    const int threadId = threadIdx.z*(blockDim.x * blockDim.y)+ threadIdx.y* blockDim.x+ threadIdx.x;
    const int blockSize =blockDim.x * blockDim.y * blockDim.z;
    
    while(1)/* this while can solve a sssp */
    {
        u = threadId;
        QuickExit = 0;
        while(u < (*n))
        {
            for(int adj = V[u]; adj < V[u+1]; adj++)
            {
                v = E[adj];
                old=atomicMin( &dist[v] , dist[u] + W[adj]);
                if(old>dist[v])
                {
                    QuickExit=1;
                    visit[v]=1;
                }
            }
            u+=blockSize;
        }
        __syncthreads();
        if(QuickExit==0){
            break;
        }
    }
}