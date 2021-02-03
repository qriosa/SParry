__global__ void delta_stepping(int* V, int* E, int* W, int* n, int* delta, int* dist, int* predist, int* B, int* hadin){
    const int u0 = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x ; 
    const int s0 = gridDim.x * blockIdx.y + blockIdx.x;
    const int offset = blockDim.x * blockDim.y * blockDim.z; // the number of threads in a block
    const int blockNum = (const int) gridDim.x * gridDim.y; // the number of blocks

	int u = -1;
	int s = -1;
	int sn = -1;
	int s0n = -1;
	int ln = (*n);

	__shared__ int id[1]; // shared id with all threads in the bucket  整个桶内共享的桶的编号
	__shared__ int maxBucketId[1]; // the max bucket id
	__shared__ int bucketNowIsNull[1]; // check bucket now is null or not

	s = s0;
	while(s < ln){ // source must valid

		sn = s * ln; // offset in the dist
		s0n = s0 * ln; // offset in the bucket

		// init
		u = u0;
		while(u < ln){
			B[s0n+ u] = -1; 
			hadin[s0n + u] = 0; 
			u += offset;
		}

		__syncthreads(); // avoid overwrite

		// init
		id[0] = 0; 
		B[s0n + s] = 0; 
		maxBucketId[0] = 0;
		bucketNowIsNull[0] = 1;  

		__syncthreads();
		
		while(1){ // the bucket id of now must less than maxBucketId 

			// while to get which vertices is in bucket.
			while(1){ // maybe enter the bucket twice or maor.

				bucketNowIsNull[0] = 0; // set it's null

				u = u0;
				while(u < ln){
					if(B[s0n + u] == id[0]){ 
						hadin[s0n + u] = 1; 
						B[s0n + u] = -1;  
							
						for(int j = V[u]; j < V[u + 1]; j++){
							if(W[j] <= (*delta)){ // light edge
								atomicMin(&predist[E[j] + sn], dist[u + sn] + W[j]);
							}
						}
					}
					u += offset;
				}
				
				__syncthreads(); 

				// check predist is updated or not
				u = u0;
				while(u < ln){
					if(predist[u + sn] < dist[u + sn]){ 

						dist[u + sn] = predist[u + sn]; 
						B[s0n + u] = dist[u + sn] / (*delta); 

						atomicMax(maxBucketId, B[s0n + u]); 
	
						if(B[s0n + u] == id[0]){ 
							bucketNowIsNull[0] = 1;
						}
					}
					u += offset;
				}
				 
				__syncthreads(); // ensure bucketNowIsNull can get into every threads.
				
				if(bucketNowIsNull[0] == 0){
					break;
				}
			}
			
			__syncthreads(); // avoid updating miss one after another.

			u = u0;
			while(u < ln){
				
				if(hadin[s0n + u]){ 
					hadin[s0n + u] = 0; // next bucket can use it again.
					
					for(int j = V[u]; j < V[u + 1]; j++){
						if(W[j] > (*delta)){ // height weight
							atomicMin(&predist[E[j] + sn], dist[u + sn] + W[j]);
						}
					}
				}
				u += offset;
			}

			__syncthreads(); // wait for all is done.

			// check predist is updated or not
			u = u0;
			while(u < ln){
				if(predist[u + sn] < dist[u + sn]){ 

					dist[u + sn] = predist[u + sn]; // update dist
					B[s0n + u] = dist[u + sn] / (*delta); // update the bucket id of now.
	
					atomicMax(maxBucketId, B[s0n + u]); // update the max id of the bucket
				}
				u += offset;
			}

			// __syncthreads();  // it's not necessary to sync, even there is a update of maxBucketId
			if(u0 == 0){
				atomicAdd_block(id, 1); // next bucket.
				bucketNowIsNull[0] = 1; // assume the there is a next vertex.
			}

			__syncthreads(); // get the right id of each thread, maxBucketId avoid tid = 0 get the break, but others update maxBucketId later. 

			if(id[0] > maxBucketId[0])
				break;
					
		}
		__syncthreads();
		s += blockNum; // next source
	}
}
