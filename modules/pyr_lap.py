import torch
from torch.autograd import Variable
import torch.nn.functional as F

def dec_lap_pyr(X,levs):
    pyr = []
    cur = X
    for i in range(levs):
        cur_x = cur.size(2)
        cur_y = cur.size(3)
        cur_z = cur.size(4)

        x_small = F.interpolate(cur, (max(cur_x//2,1), max(cur_y//2,1), max(cur_z//2,1)), mode='trilinear')
        x_back  = F.interpolate(x_small, (cur_x,cur_y,cur_z), mode='trilinear')
        lap = cur - x_back
        pyr.append(lap)
        cur = x_small

    pyr.append(cur)

    return pyr

def syn_lap_pyr(pyr):

    cur = pyr[-1]
    levs = len(pyr)
    for i in range(0,levs-1)[::-1]:
        up_x = pyr[i].size(2) 
        up_y = pyr[i].size(3) 
        up_z = pyr[i].size(4) 
        #if i == 0:
            #cur = F.interpolate(cur,(up_x,up_y,up_z), mode='trilinear')
        #else:
        cur = pyr[i] + F.interpolate(cur,(up_x,up_y,up_z), mode='trilinear')

    return cur

def syn_lap_pyr_edge(pyr):

    cur = pyr[-2]
    levs = len(pyr)
    for i in range(1,levs-1)[::-1]:
        up_x = pyr[i].size(2) * 2
        up_y = pyr[i].size(3) * 2
        up_z = pyr[i].size(4) * 2
        #if i == 0:
            #cur = F.interpolate(cur,(up_x,up_y,up_z), mode='trilinear')
        #else:
        cur = pyr[i-1] + F.interpolate(cur,(up_x,up_y,up_z), mode='trilinear')

    return cur