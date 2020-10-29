// local memory 大小有限制 localSize 大小有限 只能容纳 100 * 1024 个点的
// 但是问题不大 整这个如果图的点太多 我们直接认为是大图 启用图分割 还是可以对
// 普通 delta 
__global__ void delta_stepping(int* V, int* E, int* W, int* n, int* s, int* delta, int* dist, int* predist, int* nowIsNull, int* quickBreak){
	
	const int u0 = (const int)threadIdx.x;
	const int offset = (const int) blockDim.x;
	const int localSize = 100; // 用这个来调控开启的多少

	// nowIsNull 全局变量 用于标记当前轮的桶是否是空的 1就是非空
	// quickBreak 为1 就说明有桶有点

	int B[localSize]; // 最多支持1024 * localSize 个点 也够了吧？ u 线程的 B[i] 实际意义是 代表点 i * blockDimx.x + u 归属于哪个桶
	bool hadin[localSize]; // 同理表示某个点是否曾经在当前桶中呆过  主要用于重边
	bool vis[localSize]; // 同理 就是标记某个点是否距离发生过改变 predist 会更快
	int id = 0; // 当前处理的桶的编号
	int u = -1; // 实际参与的代表结点
	int pos = -1; // 每个线程代表的点中的第几个点

	//// 初始化 局部变量
	for(int i = 0; i < localSize; i++){
		B[i] = -1;
		hadin[i] = 0;
		vis[i] = 0;
	}

	if(u0 == (*s) % offset){
		pos = (*s) / offset;
		B[pos] = 0; // 将源点标记为在0号桶中。 
		vis[pos] = 1;
		hadin[pos] = 1;
	}

	__syncthreads();

	while(1){ // 死循环
		//// 遍历一遍 以确认所有桶均为空了

		u = u0;
		while(u < (*n) && (*quickBreak) == 0){
			if(B[u / offset] != -1){ // 这说明至少一个点还在桶中，不予退出
				atomicExch(quickBreak, 1);
				//printf(">id = %2d, u = %3d 还在 Bid = %2d\n", id, u, B[u / offset]);
				break;
			}
			u += offset;
		}

		__syncthreads();
		
		if(*quickBreak == 0){ // 0就是 所有点都空了，不存在有桶有点了  就跳出循环
			// printf("我退出了 u = %d， id = %d\n", u, id);
			break;
		}
		
		//// 以下才是一轮更新桶中做的事
		while((*nowIsNull)){

			//// 遍历每一个点判断是否在当前桶中
			u = u0;
			while(u < *n){
				
				pos = u / offset;

				if(B[pos] == id){ // 当前结点u它属于当前桶中
					B[pos] = -1; // 从当前桶中剔除
					hadin[pos] = 1; // 那么当前这个点曾经也出现过在本桶中

					//printf("-id = %2d, u = %3d 剔除 Bid = %2d\n", id, u, id);
					
					if(vis[pos]){ // 当前结点u的距离发生过变化
						vis[pos] = 0; // 标记当前结点是距离在使用之后未发生过变化了
						
						for(int j = V[u]; j < V[u + 1]; j++){ // 枚举当前结点的所有邻居结点
							if(W[j] <= (*delta)){ // 轻边
								//printf("id = %d, 结点 u = %d 有这些轻边连接点: %d\n", id, u, E[j]);
								atomicMin(&predist[E[j]], dist[u] + W[j]);
							}
						}
					}
				} 
				u += offset;
			}

			//// 如果在一开进while就设置为空的话 有可能跑得快的先进来设置为0了，跑得慢的一看是0，就不进来了，就错了
			*nowIsNull = 0; // 标记当前桶空的 
			__syncthreads();

			//// 检测 predist 判断是否有点在本轮被更新了 同时检测是否有新点落入当前桶中
			u = u0;
			while(u < (*n)){
				if(predist[u] < dist[u]){ // 如果当前结点 predist 更小了，说明其可以被更新了
					
					pos = u / offset; // 计算的u相对偏移

					dist[u] = predist[u]; // 更新其 dist
					B[pos] = dist[u] / (*delta); // 计算更新后应该属于哪个桶
					vis[pos] = 1; // 标记其距离是发生过变化了
					//printf("+id = %2d, u = %3d 加入 Bid = %2d\n", id, u, B[pos]);

					if(B[pos] == id){ // 如果更新后还是落入当前桶中 那么就说明当前桶中是还有有点的
						// 当前桶中还有点 标记当前桶不是空的
						*nowIsNull = 1;
					}
				}
				u += offset;
			}
			__syncthreads();
		}

		//// 开始处理重边
		u = u0;
		while(u < (*n)){
			pos = u / offset;
			if(hadin[pos]){ // 当前结点在本轮中进入过当前桶中 进入过当前桶其距离是一定发生过改变的  所以不要vis来判断
				hadin[pos] = 0; // 标记为没有进入过了 下一个桶可以继续用
				
				for(int j = V[u]; j < V[u + 1]; j++){
					if(W[j] > (*delta)){ // 重边
						//printf("id = %d, 结点 u = %d 有这些重边连接点: %d\n", id, u, E[j]);
						atomicMin(&predist[E[j]], dist[u] + W[j]);
					}
				}
			}
			u += offset;
		}
		__syncthreads();

		//// 检测 predist 判断是否有点在本轮被更新了 同时检测是否有新点落入当前桶中
		u = u0;
		while(u < (*n)){
			if(predist[u] < dist[u]){ // 如果当前结点 predist 更小了，说明其可以被更新了
				
				pos = u / offset; // 计算的u相对偏移

				dist[u] = predist[u]; // 更新其 dist
				B[pos] = dist[u] / (*delta); // 计算更新后应该属于哪个桶
				vis[pos] = 1; // 标记其距离是发生过变化了
				
				//printf("+id = %d, u = %d 加入 Bid = %d\n", id, u, B[pos]);					
				
				// 由重边产生的更新不可能再落入当前桶中
				// if(B[pos] == id){ // 如果更新后还是落入当前桶中 那么就说明当前桶中是还有有点的
				// 	// 当前桶中还有点
				// 	标记当前桶不是空的
				// }
			}
			u += offset;
		}

		id += 1; // 进入下一个桶
		*nowIsNull = 1; // 假设下一轮的桶中有点
		*quickBreak = 0; // 假设所有桶都空了
		__syncthreads();
	}	
}