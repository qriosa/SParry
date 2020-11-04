__global__ void kernelForAPSP(int *V, int *E, int *W, int *n, int *dist, int *predist){
    /*thread 和 block 的两级复用 ，暂时不使用dist矩阵对称特性的利用，也不开启随机计算优化*/
    const int blockId  = blockIdx.z *(gridDim.x *  gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    const int threadId = threadIdx.z*(blockDim.x * blockDim.y)+ threadIdx.y* blockDim.x+ threadIdx.x;
    const int blockSize =blockDim.x * blockDim.y * blockDim.z;
    const int gridSize = gridDim.x  * gridDim.y  * gridDim.z;
    __shared__ int QuickExit;
    __shared__ bool visit[6000];
    int u=0, st=0, align=0, old=0;
    if(blockId >= (*n)) return ;
    if(threadId >= (*n)) return ;
    st = blockId;
    memset(visit,false,sizeof(visit));
    while(st < (*n))
    {
        // memset(visit,false,sizeof(visit));
        visit[st]=1;
        __syncthreads();
        align = (st * (*n));
        // while(1){/*这个while里解决了一个单元最短路问题*/
        for(int rnd=0;rnd<(*n);rnd++){
            QuickExit = 0;
            u = threadId;
            while(u < (*n)){
                if(visit[u]){
                    visit[u]=0;
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
                    visit[u] = 1;
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
        // /*这里开始dist中间结果利用*/
        // u=threadId;
        // while(u < (*n)){
        //     int ualign = u * (*n);
        //     old=atomicMin(&dist[ualign + st],dist[align + u]);
        //     if(old > dist[ualign + st]){
        //         visit[st]=1;
        //     }
        //     u+=blockSize;
        // }
        st += gridSize;
    }
}


// __global__ void kernelForAPSP(int *V, int *E, int *W, int *n, bool *visit, int *dist, int *predist){
//     /*thread 和 block 的两级复用 ，暂时不使用dist矩阵对称特性的利用，也不开启随机计算优化*/
//     const int blockId  = blockIdx.z *(gridDim.x *  gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
//     const int threadId = threadIdx.z*(blockDim.x * blockDim.y)+ threadIdx.y* blockDim.x+ threadIdx.x;
//     const int blockSize =blockDim.x * blockDim.y * blockDim.z;
//     const int gridSize = gridDim.x  * gridDim.y  * gridDim.z;
//     __shared__ int QuickExit;
//     int u=0, st=0, align=0, old=0;
//     // if(blockId >= (*n)) return ;
//     // if(threadId >= (*n)) return ;
//     st = blockId;
//     while(st < (*n))
//     {
//         align = (st * (*n));
//         // while(1){/*这个while里解决了一个单元最短路问题*/
//         for(int rnd=0;rnd<(*n);rnd++){
//             QuickExit = 0;
//             u = threadId;
//             while(u < (*n)){
//                 if(visit[u + align]){
//                     visit[u + align]=0;
//                     for(int adj = V[u];adj<V[u+1];adj++){
//                         old=atomicMin( &predist[align + E[adj]] , dist[align + u] + W[adj]);
//                     }
//                 }
//                 u+=blockSize;
//             }
//             __syncthreads();
//             u=threadId;
//             while(u < (*n)){
//                 if(predist[align + u] < dist[align + u]){
//                     dist[align + u] = predist[align + u];
//                     visit[align + u] = 1;
//                     QuickExit = 1;
//                 }
//                 u+=blockSize;
//             }
//             __syncthreads();
//             if(QuickExit==0){
//                 break;
//             }
//         }
//         // __syncthreads();
//         /*这里开始dist中间结果利用*/
//         u=threadId;
//         while(u < (*n)){
//             int ualign = u * (*n);
//             old=atomicMin(&dist[ualign + st],dist[align + u]);
//             if(old > dist[ualign + st]){
//                 visit[ualign + st]=1;
//             }
//             u+=blockSize;
//         }
//         st += gridSize;
//     }
// }