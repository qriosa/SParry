__global__ void dijkstra(int* V, int* E, int* W, int* n, int* vis, int* dist, int* predist){
	const int u0 = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; 
	const int offset = blockDim.x * blockDim.y * blockDim.z; 
    __shared__ int quickBreak[1];
    
    int u = -1;

	for(int i = 0; i < (*n); i++){

		quickBreak[0] = 0;
		
		u = u0;
		while(u < *n){ 
			if(vis[u] == 0){ 
				vis[u] = 1;
				for(int j = V[u]; j < V[u + 1]; j++){
					atomicMin(&predist[E[j]], dist[u] + W[j]);
				}
			}
			u += offset;
		}
		__syncthreads(); 

		u = u0;
		while(u < *n){
			if(predist[u] < dist[u]){ 
				dist[u] = predist[u];
				vis[u] = 0; 
				quickBreak[0] = 1;
			}
			u += offset;
		}
		__syncthreads(); 
		if(quickBreak[0] == 0)
			break;
		__syncthreads(); 
	}	
}

__global__ void divide(int* V, int* E, int* W, int* n, int* flag, int* base, int* part, int* vis, int* dist, int* predist){
	
	const int u0 = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; 
	const int offset = blockDim.x * blockDim.y * blockDim.z; 
	
	int u = -1;
	int l = -1;
	int r = -1;
	int localBase = base[0];
	int localPart = part[0];
	
	u = u0; //u + l is the vertex index
	while(u < (*n)){ 

		if(V[u + 1] <= localBase){ // self right
			u += offset;
			continue; // illegal
		}
		else if(V[u] >= localBase + localPart){ // self left
			u += offset;
			continue; // illegal
		}

		if(vis[u]){ 
			atomicSub(&vis[u], 1);

			l = localBase>V[u]?localBase:V[u];
			r = (localBase + localPart)<V[u + 1]?(localBase + localPart):V[u + 1];
			
			for(int j = l; j < r; j++){ 
				atomicMin(&predist[E[j - localBase]], dist[u] + W[j - localBase]); 
			}
		}
		
		u += offset;
	}
	__syncthreads(); 

	u = u0;
	while(u < (*n)){
		if(predist[u] < dist[u]){ 
			dist[u] = predist[u];
			vis[u] = (V[u + 1] + localPart - 1) / localPart - V[u] / localPart; // re calc the ability of updating 
			flag[0] = 1;
		}
		u += offset;
	}
}
