from scipy.spatial import distance
import collections
import sklearn
import pandas as pd
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

## ============= Convert follicle object ============= ##
""" The Follicle object doesnt make sense, I'm changing it to a 'slice dict'"""
def convert_fol_dict(fd):
    sd = collections.defaultdict()
    # init the sd dict
    for slice in fd[1].bbox.iterkeys():
        sd[int(slice)] = collections.defaultdict()

    for id,fol in fd.iteritems():
        for slice in fol.bbox.iterkeys():
            sd[int(slice)][id] = {}
            if slice in fol.bbox.keys():
                sd[int(slice)][id]['bbox'] = fol.bbox[slice]
            if slice in fol.inner.keys():
                sd[int(slice)][id]['inner'] = fol.inner[slice]
            if slice in fol.outer.keys():
                sd[int(slice)][id]['outer'] = fol.outer[slice]
            if slice in fol.centroid.keys():
                sd[int(slice)][id]['centroid'] = fol.centroid[slice]
    return(sd)


def convert_fol_dict_file(fol_file):
    fd = load_fol_data(fol_file)
    slice_dict_name = os.path.splitext(fol_file)[0]+'_sd.pckl'
    sd = convert_fol_dict(fd)
    with open(slice_dict_name,'w') as fid:
        pickle.dump(sd,fid)
    print('Wrote slice dict to {}'.format(slice_dict_name))


def sd_to_centroids():

    def cost(current,next):
        D = distance.cdist(current[['x','y']].as_matrix(),next[['x','y']].as_matrix())
        D = np.min(D,axis=1)


    temp = []
    for slice_num, slice in sd.iteritems():
        for id,fol in slice.iteritems():
            if 'centroid' in fol.keys():

                temp.append([slice_num,id,fol['centroid'][0],fol['centroid'][1]])

    df_centroid = pd.DataFrame(data=temp,columns=['slice_num','id','x','y'])
    X = df_centroid[['x','y','slice_num']].as_matrix()
    for slice in df_centroid['slice_num'].unique():
        current = df_centroid.loc[df_centroid.slice_num==slice,['x','y']]
        next = df_centroid.loc[df_centroid.slice_num==slice+1,['x','y']]
















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





















