import numpy as np

def patch_sar(x,patch_r):
    H, W=x.shape
    x = np.pad(x, (patch_r, patch_r), "constant", constant_values=0)
    patch_size=2*patch_r+1
    p=[]
    for i in range(H):
        for j in range(W):
            u=(x[i:i+patch_size,j:j+patch_size]).flatten()
            p.append(u)
    return np.array(p)
def patch_rgb(x,patch_r):
    H, W,C=x.shape
    x1=[]
    for c in range(C):
        xx = np.pad(x[:,:,c], (patch_r, patch_r), "constant", constant_values=0)
        x1.append(xx)
    patch_size=2*patch_r+1
    p=[]
    x1=np.array(x1)
    pos=[]
    for i in range(H):
        for j in range(W):
            u=(x1[:,i:i+patch_size,j:j+patch_size]).flatten().tolist()
            p.append(u)
            pos.append([i,j])

    return np.array(p),pos