# -*- coding: utf-8 -*-

import numpy as np
from time import time
from numba import cuda

@cuda.jit('void(float64[:],float64[:],int32,int32,float64[:,:])')
def grid_search_kernel(alpha_grid, beta_grid, n_alpha, n_beta, costs):
    i,j = cuda.grid(2)

    if i < n_alpha and j < n_beta:
        costs[i,j] = alpha_grid[i]+beta_grid[j]
        
def grid_search_parallel(alpha_grid, beta_grid):
    n_alpha = alpha_grid.shape[0]
    n_beta = beta_grid.shape[0]
    
    d_alpha_grid = cuda.to_device(alpha_grid)
    d_beta_grid = cuda.to_device(beta_grid)
    d_costs = cuda.device_array(shape = [n_alpha, n_beta], dtype = np.float64)
    
    TPBX, TPBY = 16, 16
    block_dims = TPBX, TPBY
    grid_dims = (n_alpha+TPBX-1)//TPBX, (n_beta+TPBY-1)//TPBY
    grid_search_kernel[grid_dims, block_dims](d_alpha_grid, d_beta_grid, n_alpha, n_beta, d_costs)

    costs = d_costs.copy_to_host()
    optimal_ind = np.unravel_index(np.argmin(costs, axis=None), costs.shape)
    print(optimal_ind)
    return alpha_grid[optimal_ind[0]], beta_grid[optimal_ind[1]]

if __name__ == "__main__":
    t0 = time()
    alpha_grid = np.linspace(0,10,num=1000)
    beta_grid = np.linspace(0,10,num=1000)
    alpha, beta = grid_search_parallel(alpha_grid, beta_grid)
    print('Optimal alpha:', alpha)
    print('Optimal beta:', beta)
    print('Run Time:', time()-t0)
