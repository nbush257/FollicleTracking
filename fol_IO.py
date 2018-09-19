import collections
from scipy.spatial import distance
from auto_fol_finder import Follicle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import cPickle as pickle
from skimage import io
import re
from skimage import draw
from mpl_toolkits import mplot3d

def load_fol_data(fol_data_file):
    with open(fol_data_file,'r') as fid:
        fol_dict = pickle.load(fid)
    return(fol_dict)




def mask_to_patch_coords(r_min,c_min,r_max,c_max):

   h  =r_max-r_min
   w = c_max-c_min
   coord = [c_min,r_min]
   return(coord,h,w)


def plot_all_fols(fol_dict,I_file):
    I = io.imread(I_file)
    slice_num = re.search('_\d{4}\.',I_file).group()[1:-1]
    plt.imshow(I,'gray')
    for fol in fol_dict.itervalues():
        try:
            inner = fol.inner[slice_num]
            outer = fol.outer[slice_num]
        except KeyError:
            # If there isn't a follicle for this key, just skip it
            continue
        if len(inner)==0:
            continue
        plt.plot(inner[1],inner[0],'r.',markersize=1)
        plt.plot(outer[1],outer[0],'g.',markersize=1)

def plot_all_bboxes(fol_dict,I_file):
    I = io.imread(I_file)
    slice_num = re.search('_\d{4}\.',I_file).group()[1:-1]
    plt.imshow(I,'gray')
    for fol in fol_dict.itervalues():
        try:
            bbox = fol.bbox[slice_num]
        except KeyError:
            # If there isn't a follicle for this key, just skip it
            continue
        if len(bbox)==0:
            continue

        coord,h,w = mask_to_patch_coords(*bbox)
        rect = patches.Rectangle(coord,w,h,facecolor='none',edgecolor='r',linewidth=3)
        plt.gca().add_patch(rect)


## ============== Alignment code needs work ======= #

def align_fols(fd):
    C = collections.defaultdict()
    all_centroids = []
    all_id = []
    for id,fol in fd.iteritems():
        for slice_num in fol.centroid:
            if slice_num not in C.keys():
                C[slice_num] = []

            else:
                C[slice_num].append(fol.centroid[slice_num])
    for slice_num,centroids in C.iteritems():
        centroids = np.array(centroids)
        slice_vector = np.ones(centroids.shape[0])*int(slice_num)
        centroids = np.concatenate([centroids,slice_vector[:,np.newaxis]],axis=1)
        all_centroids.append(centroids)
        all_id.append(range(centroids.shape[0]))

    all_centroids = np.concatenate(all_centroids)
    all_id = np.concatenate(all_id)
    return(all_centroids,all_id)

def map_centroids(centroids,id):
    slices = np.array(sorted(list(set(centroids[:,-1]))),dtype='int')
    fol_map = {}

    mask = centroids[:,-1]==slices[0]
    fol_map[slices[0]] = np.array(range(np.sum(mask)))

    for idx in slices:
        mask = centroids[:,-1]==idx
        c1 = centroids[mask,:2]
        mask = centroids[:,-1]==idx+1
        c2 = centroids[mask,:2]
        D = distance.cdist(c1,c2)
        fol_map[idx+1] = np.argmin(D,axis=1)





















