import torch
import torch.nn as nn
import numpy as np
class SVD(nn.Module):
    def __init__(self, rgb = False):
        super().__init__()
        self.rgb = rgb
    
        #SVD = UEVt, where U is mxm, E is mxn, Vt is nxn, to get an mxn
    def forward(self,  image, rank_approx = 1): #feed in image, get the rank_approx versiono f that image
        #image should be a matrix, m x n 
        #if rgb, should be 3 x m x n
        m = image.shape[0]
        n = image.shape[1]
        ata = image.T @ image # n x m * m x n (total nxn matrix, columns of V are the eigenvecotrs)
        aat = image @ image.T # mx n * n xm (total mxm matrix, eigenvectors = U), not actually needed
        eigenvalues, v_cols = np.linalg.eigh(ata) #eigenvectors = V
        #Avi = singularvalue i * mui
        #Mui = Avi / singular values
        idx = np.argsort(eigenvalues)[::-1] 
        eigenvalues = eigenvalues[idx]
        v_cols = v_cols[:, idx]
        singular_values = np.sqrt(np.maximum(eigenvalues, 0))
        u_cols = (image @ v_cols) / singular_values
        rank_i_approx = torch.zeros((m, n))
        for i in range(rank_approx):
            rank_i_approx += singular_values[i] * np.outer(u_cols[:, i],v_cols[:, i])
        return rank_i_approx #should be an mx n version of the matrix






