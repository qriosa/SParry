// 使用了 shared memory 的矩阵相乘

__global__ void vectorAdd_MinSharedMemory(int* a, int* b, int* c, int* INF_global){
    const int blockSize = 16; // 共享内存的参数 确定共享内存的大小 
    __shared__ int sharedA[blockSize*blockSize];
    __shared__ int sharedB[blockSize*blockSize];
    
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int col = blockDim.y * blockIdx.y + threadIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    int INF = (*INF_global);//初始化为无穷大
    int tempShare;
    
    if(row>=*b||col>=*b)
        return;
    
        for(int k = 0 ; k < (int)((*b+blockSize-1)/blockSize); k++){
            sharedA[tx*blockSize+ty] = a[row*(*b)+ty+k*blockSize];
            sharedB[tx*blockSize+ty] = a[(tx+k*blockSize)*(*b)+col];
        
            __syncthreads();
        
            for(int kk = 0 ; kk < blockSize ; kk++){
                tempShare = sharedA[tx*blockSize+kk]+sharedB[kk*blockSize+ty];
                
                if(tempShare<0)
                    continue;
                
                INF = (INF < tempShare)?INF:tempShare;
            }
            __syncthreads();
    }
    c[row*(*b)+col] = INF;
}