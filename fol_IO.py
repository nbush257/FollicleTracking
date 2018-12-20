from scipy.spatial import distance
import scipy.io.matlab as sio
import os
import collections
import sklearn
import pandas as pd
from scipy.spatial import distance
# from auto_fol_finder import Follicle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import cPickle as pickle
from skimage import io
import re
from skimage import draw
from mpl_toolkits import mplot3d


def load_fol_data(fol_data_file):
    '''
    Wrapper that loads a slice dictionary in easily
    :param fol_data_file: file to load
    :return: data_dict -- The dictionary containing the tracked points of all the slices in a stack
    '''
    with open(fol_data_file,'r') as fid:
        data_dict = pickle.load(fid)
    return(data_dict)


def verts_to_patch_coords(r_min,c_min,r_max,c_max):
    """
    Converts the 4-length bounding box points (bottom left, top right) into
    a coord,h,w format used by matplotlib patches
    :param r_min:
    :param c_min:
    :param r_max:
    :param c_max:
    :return:
    """
    h  =r_max-r_min
    w = c_max-c_min
    coord = [c_min,r_min]
    return(coord,h,w)

def test_sub_dict(sd_file):
    """
    Create a subset of the data that is easier to work with, taking the first three slices of the
    dictionary
    :param sd_file: The full tracked pickle file
    :return: None, saves a pickle file as test.pckl
    """
    dat = load_fol_data(sd_file)
    temp = {}
    temp[1] = dat[1]
    temp[2] = dat[2]
    temp[3] = dat[3]
    path = os.path.split(sd_file)[0]
    with open(path+'\\test.pckl','w') as fid:
        pickle.dump(temp,fid)


def convert_to_h5():
    pass

def convert2mat(sd_file):
    pass
    dat = load_fol_data(sd_file)
    outfilename = os.path.splitext(sd_file)[0]+'.mat'
    sio.savemat(outfilename,dat)

def plot_all_fols(slice_dict,I_file):
    """
    Load an image and plot all the follicles. Seems buggy
    :param slice_dict: The tracked follicles in the slice_dict format
    :param I_file: An image file
    :return: None
    """
    I = io.imread(I_file)
    slice_num = int(re.search('_\d{4}\.',I_file).group()[1:-1])
    plt.imshow(I,'gray')
    slice = slice_dict[slice_num]
    for fol in slice.itervalues():
        inner = fol['inner']
        outer = fol['outer']
        if inner:
            plt.plot(inner[1],inner[0],'r.',markersize=1)
        if outer:
            plt.plot(outer[1],outer[0],'g.',markersize=1)

def plot_all_bboxes(slice_dict,I_file):

    """
    Load an image and plot all the bounding boxes.
    :param slice_dict: The tracked follicles in the slice_dict format
    :param I_file: An image file
    :return: None
    """
    I = io.imread(I_file)
    slice_num = re.search('\d{4}\.',I_file).group()[1:-1]
    plt.imshow(I,'gray')
    slice = slice_dict[int(slice_num)]
    for fol in slice.itervalues():
        bbox = fol['bbox']
        coord,h,w = verts_to_patch_coords(*bbox)
        rect = patches.Rectangle(coord,w,h,facecolor='none',edgecolor='r',linewidth=3)
        plt.gca().add_patch(rect)
    plt.show()

def convert_fol_dict(fd):
    """
    Convert a fol_dict into a slice_dict
    :param fd: a fol_dict dicitonary as it is spit out by the tracking code
    :return: sd - the same data reshaped into a slice_dict.
    """
    sd = collections.defaultdict()
    # init the sd dict
    for slice in fd[1].bbox.iterkeys():
        sd[int(slice)] = collections.defaultdict()

    for id,fol in fd.iteritems():
        for slice in fol.bbox.iterkeys():
            sd[int(slice)][id] = {}
            if slice in fol.bbox.keys():
                sd[int(slice)][id]['bbox'] = np.array(fol.bbox[slice])
            if slice in fol.inner.keys():
                sd[int(slice)][id]['inner'] = np.array(fol.inner[slice])
            if slice in fol.outer.keys():
                sd[int(slice)][id]['outer'] = np.array(fol.outer[slice])
            if slice in fol.centroid.keys():
                sd[int(slice)][id]['centroid'] = np.array(fol.centroid[slice])
    return(sd)


def convert_fol_dict_file(fol_file):
    """
    wrapper to convert fol_dict which takes a fol_dict pickle file and saves
    a complementary slice_dict pickle file
    :param fol_file: Filename of the fol_dict pickle file
    :return: None, saves the slice_dict pickle file
    """
    fd = load_fol_data(fol_file)
    slice_dict_name = os.path.splitext(fol_file)[0]+'_sd.pckl'
    sd = convert_fol_dict(fd)
    with open(slice_dict_name,'w') as fid:
        pickle.dump(sd,fid)
    print('Wrote slice dict to {}'.format(slice_dict_name))

### ========================= ###
#   EVERYTHING BELOW HERE IS UNFINISHED. YOU MAY FIND BITS AND PIECES
#   THAT ARE USEFUL IN BUILDING MORE ALIGNMENT/LABELLING CODE
### ========================= ###

def sd_to_centroids():
    # UNFINISHED
    """
    the goal of this function was to start aligning and labelling the centroids
    :return:
    """

    pass
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
    # Supposed to add ids to the follicles
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
    # supposed to map arbitrary follicle ids to a labeled follicle ID.
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





















