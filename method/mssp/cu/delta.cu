__global__ void delta_stepping(int* V, int* E, int* W, int* n, int* srcNum, int* srclist, int* delta, int* dist, int* predist){

    const int u0 = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    const int s0 = gridDim.x * blockIdx.y + blockIdx.x;
    const int offset = blockDim.x * blockDim.y * blockDim.z; // the number of threads in a block.
    const int blockNum = gridDim.x * gridDim.y; // the number of block
    const int localSize = 100;

    // nowIsNull, global var, indicate whether the bucket is empty or not, 1 is not.
    // quickBreak 1 indicate the bucket has vertices.  

    int B[localSize]; // max support 102400 vertices. B[i] in thread u is the vertex i * blockDimx.x + u is belong to which bucket. 
    bool hadin[localSize]; // indicate one vertex is in or once in a bucket.
    bool vis[localSize]; 
    int id; // the bucket id
    int u = -1; // the vertex
    int pos = -1; // the index of vertices will be calc by the thread.
    int sIndex = -1;
    int sn = -1;

    __shared__ int nowIsNull[1];
    __shared__ int quickBreak[1];   

    sIndex = s0;
    while(sIndex < (*srcNum)){
        
        // init
        id = 0;

        for(int i = 0; i < localSize; i++){
            B[i] = -1;
            hadin[i] = 0;
            vis[i] = 0;
        }

        if(u0 == srclist[sIndex] % offset){
            pos = srclist[sIndex] / offset;
            B[pos] = 0; // put source vertex into bucket 0.
            vis[pos] = 1;
            hadin[pos] = 1;
        }

        nowIsNull[0] = 1;
        quickBreak[0] = 1;

        sn = sIndex * (*n);  

        __syncthreads();    

        while(1){ 
            u = u0;
            while(u < (*n) && quickBreak[0] == 0){
                if(B[u / offset] != -1){ // at least one vertex is in bucket.
                    quickBreak[0] = 1;
                    break;
                }
                u += offset;
            }

            __syncthreads();
            
            if(quickBreak[0] == 0){ // 0 is all bucket is empty.
                break;
            }
            
            // update bucket
            while(nowIsNull[0]){

                // iterate every vertices, and tell it is in this bucket or not.
                u = u0;
                while(u < *n){
                    
                    pos = u / offset;

                    if(B[pos] == id){ // u is belong to the current bucket
                        B[pos] = -1; // delete from current bucket
                        hadin[pos] = 1; 
                        
                        if(vis[pos]){ // the distence of the vertex u has been changed
                            vis[pos] = 0;                          
                            for(int j = V[u]; j < V[u + 1]; j++){ // iterate all neibor
                                if(W[j] <= (*delta)){ // light edge
                                    atomicMin(&predist[E[j] + sn], dist[u + sn] + W[j]);
                                }
                            }
                        }
                    } 
                    u += offset;
                }

                nowIsNull[0] = 0; // set current bucket is empty
                __syncthreads();

                u = u0;
                while(u < (*n)){
                    if(predist[u + sn] < dist[u + sn]){ // if predist is less, then dist can be update
                        
                        pos = u / offset; // calc the offset

                        dist[u + sn] = predist[u + sn]; // update the dist
                        B[pos] = dist[u + sn] / (*delta); // calc after updated should goto which bucket                      vis[pos] = 1; // 标记其距离是发生过变化了

                        if(B[pos] == id){ 
                            // set current bucket is not empty
                            nowIsNull[0] = 1;
                        }
                    }
                    u += offset;
                }
                __syncthreads();
            }


            // heavy edge
            u = u0;
            while(u < (*n)){
                pos = u / offset;
                if(hadin[pos]){ 
                    hadin[pos] = 0; // set as 0, indicate this vertex to next bucket
                    
                    for(int j = V[u]; j < V[u + 1]; j++){
                        if(W[j] > (*delta)){ // heavy edge
                            atomicMin(&predist[E[j] + sn], dist[u + sn] + W[j]);
                        }
                    }
                }
                u += offset;
            }
            __syncthreads();

            u = u0;
            while(u < (*n)){
                if(predist[u + sn] < dist[u + sn]){ 
                    
                    pos = u / offset; 

                    dist[u + sn] = predist[u + sn]; 
                    B[pos] = dist[u + sn] / (*delta); 
                    vis[pos] = 1; 
                }
                u += offset;
            }

            id += 1; // goto next bucket
            nowIsNull[0] = 1; // assume the next bucket has vertices
            quickBreak[0] = 0;
            __syncthreads();
        }   
        sIndex += blockNum;
    }
}