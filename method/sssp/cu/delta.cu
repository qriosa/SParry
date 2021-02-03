__global__ void delta_stepping(int* V, int* E, int* W, int* n, int* s, int* delta, int* dist, int* predist, int* nowIsNull, int* quickBreak){
	
	const int u0 = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;  
	const int offset = blockDim.x * blockDim.y * blockDim.z; 
	const int localSize = 100; // malloc how many bucket

	int B[localSize]; 
	bool hadin[localSize]; 
	bool vis[localSize]; 
	int id = 0; // the bucket id
	int u = -1; 
	int pos = -1; // which point of the points each thread represents

	// init
	for(int i = 0; i < localSize; i++){
		B[i] = -1;
		hadin[i] = 0;
		vis[i] = 0;
	}

	if(u0 == (*s) % offset){
		pos = (*s) / offset;
		B[pos] = 0; // put source vertex into bucket 0
		vis[pos] = 1;
		hadin[pos] = 1;
	}

	__syncthreads();

	while(1){

		u = u0;
		while(u < (*n) && (*quickBreak) == 0){
			if(B[u / offset] != -1){ // at least one vertex is in bucket.
				atomicExch(quickBreak, 1);
				break;
			}
			u += offset;
		}

		__syncthreads();
		
		if(*quickBreak == 0){ 
			break;
		}
		
		
		while((*nowIsNull)){

			u = u0;
			while(u < *n){
				
				pos = u / offset;

				if(B[pos] == id){ 
					B[pos] = -1; 
					hadin[pos] = 1; 
					
					if(vis[pos]){ // tell the dist of the vertex is changed or not 
						vis[pos] = 0; 
						
						for(int j = V[u]; j < V[u + 1]; j++){ 
							if(W[j] <= (*delta)){ // light edge
								atomicMin(&predist[E[j]], dist[u] + W[j]);
							}
						}
					}
				} 
				u += offset;
			}

			*nowIsNull = 0; // set current bucket is empty
			__syncthreads();

			u = u0;
			while(u < (*n)){
				if(predist[u] < dist[u]){ 
					
					pos = u / offset; 

					dist[u] = predist[u]; // update dist
					B[pos] = dist[u] / (*delta); // calc after updating, it should be put into which bucket
					vis[pos] = 1; 

					if(B[pos] == id){ 
						// current bucket is not empty
						*nowIsNull = 1;
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
				hadin[pos] = 0; 
				
				for(int j = V[u]; j < V[u + 1]; j++){
					if(W[j] > (*delta)){ // heavy edge
						atomicMin(&predist[E[j]], dist[u] + W[j]);
					}
				}
			}
			u += offset;
		}
		__syncthreads();

		u = u0;
		while(u < (*n)){
			if(predist[u] < dist[u]){ 
				
				pos = u / offset; // calc offset

				dist[u] = predist[u]; // update dist
				B[pos] = dist[u] / (*delta); // calc it should belong to which buket after updating.
				vis[pos] = 1; // record it's updated
				
			}
			u += offset;
		}

		id += 1; // enter to next bucket
		*nowIsNull = 1; // assume the next bucket has vertex
		*quickBreak = 0; 
		__syncthreads();
	}	
}