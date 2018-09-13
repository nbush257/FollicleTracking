from skimage import transform,draw,feature,io,filters,segmentation,color

from skimage.draw import ellipse_perimeter
from matplotlib import patches
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import closing,square,disk,label
from skimage.measure import regionprops


class Follicle():
    def __init__(self,ID):
        self.ID = ID
        self.whisker=''
        self.inner = defaultdict()
        self.outer = defaultdict()
        self.bbox = defaultdict()

    def add_inner(self,slice_num,inner_pts):
        if slice_num in self.keys():
            print('Overwriting previous tracking on slice {}'.format(slice_num))
        else:
            self.inner[slice_num] = inner_pts


    def add_outer(self,slice_num,outer_pts):
        if slice_num in self.keys():
            print('Overwriting previous tracking on slice {}'.format(slice_num))
        else:
            self.outer[slice_num] = outer_pts


    def add_bbox(self,slice_num,box):
        if slice_num in self.keys():
            print('Overwriting previous tracking on slice {}'.format(slice_num))
        else:
            self.inner[slice_num] = box


def ginput2boundingbox(a,b,mode='mask'):
    pts = np.array([[a[0],a[1]],[b[0],b[1]],[a[0],b[1]],[b[0],a[1]]],dtype='int')
    bot_x = np.min(pts[:,0])
    bot_y = np.min(pts[:,1])
    top_x = np.max(pts[:,0])
    top_y = np.max(pts[:,1])

    if mode=='mask':
        rr,cc = draw.rectangle((bot_y,bot_x),end=(top_y,top_x))
    elif mode=='verts':
        rr = np.array([bot_x,bot_x,top_x,top_x])
        cc = np.array([bot_y,top_y,top_y,bot_y])
    elif mode == 'patch':
        coord = (bot_x,bot_y)
        w = top_x-bot_x
        h = top_y-bot_y
        return(coord,h,w)
    else:
        raise ValueError("Unknown mode of output requested. Must be 'mask' or 'verts'")

    return(rr,cc)

def ui_major_bounding_box(I):
    while True:
        fig,ax = plt.subplots(1,2)
        plt.sca(ax[0])
        plt.imshow(I,'gray')
        plt.title('Click corners around all follicles')
        plt.draw()
        a,b= plt.ginput(n=2,timeout=0,show_clicks=True)
        rr,cc = ginput2boundingbox(a,b)
        plt.sca(ax[1])
        plt.imshow(I[rr,cc],'gray')
        plt.title('If good press enter, if not, click on new region')
        break # edit this to allow the user to redo.

    plt.close('all')
    return(rr,cc)
def zoom_to_box(ax,box):
    ax.set_xlim(np.min(box[1]),np.max(box[1]))
    ax.set_ylim(np.min(box[0]),np.max(box[0]))
    ax.invert_yaxis()


def user_get_fol_bounds(I,major_box,fol_dict=None):
    """
    Asks the user to pick out bounding boxes on individual follicles
    This should only be run at the beginning -- Needs development
    :param I: the full image of the slice
    :param major_box: the (rr,cc) of the bounding box around all follicles
    :return: dict of bounding boxes
    """
    plt.imshow(I,'gray')
    ax = plt.gca()
    zoom_to_box(ax,major_box)
    count=0
    if fol_dict is None:
        print('initializing a follicle dictionary')
        fol_dict=defaultdict()
    while True:
        count+=1
        plt.title('Click on follicle {}; right click to remove, middle click to exit completely'.format(count))
        plt.pause(0.01)
        input = plt.ginput(2,timeout=0,show_clicks=True)
        if len(input)<2:
            break
        else:
            a,b = input
        box = ginput2boundingbox(a,b,'mask')
        coord,h,w=ginput2boundingbox(a,b,'patch')
        rect = patches.Rectangle(coord,w,h,linewidth=2,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        ax.text(coord[0],coord[1],'{}'.format(count),color='c',fontsize=16)
        F = Follicle(count)
        F.bbox = box
        fol_dict[count] = F
    plt.close('all')
    return(fol_dict)



def extract_all_ellipses(I,rr,cc):
    """
    this code is intended to extract ellipses on a very large image. Probably not a good idea'
    :param I:
    :param rr:
    :param cc:
    :return:
    """
    # I_filt = filters.median(I,np.ones([9,9]))
    pass
    edges = feature.canny(I[rr,cc],sigma=3.0)

    result = transform.hough_ellipse(edges,
                                     accuracy=20,
                                     threshold=250,
                                     min_size=100,
                                     max_size=120)
    result.sort(order='accumulator')
    best = list(result[-1])
    yc,xc,a,b = [int(round(x)) for x in best[1:5]]
    orientation=best[5]

    cy,cx = ellipse_perimeter(yc,xc,a,b,orientation)

def plot_bbox_from_region(region,ax=None):
    if ax is None:
        ax = plt.gca()
    bbox = region.bbox
    coord = bbox[1::-1]
    h = bbox[2] - bbox[0]
    w = bbox[3] - bbox[1]
    rect = patches.Rectangle(coord, w, h, linewidth=5, edgecolor='r', facecolor='none')
    ax.add_patch(rect)


def extract_mask(I_sub):
    """
    Find ellipses in the follicle ROI
    :param I_sub: Sub image of just the follicel ROI
    :return:
    """
    T = filters.threshold_yen(I_sub)
    bw = I_sub<T
    bw = closing(bw,disk(9))
    region_labels = label(bw)
    props = regionprops(region_labels)
    largest_idx = np.argsort([r.area for r in props])[-1]
    mask = region_labels==np.array(largest_idx+1,dtype='int64')
    largest_region = props[largest_idx]
    region_labels[np.logical_not(mask)]=0
    boundary = segmentation.find_boundaries(region_labels,mode='inner')
    boundaries = label(boundary)
    bound_props = regionprops(boundaries)
    boundary_order = np.argsort([r.area for r in bound_props])+1
    if len(boundary_order)==1:
        outer = boundaries==boundary_order[0]
        inner = outer
        bbox = bound_props[boundary_order[0]].bbox
        print('Inner and outer follicle are indistinguishable')
    else:
        outer = boundaries==boundary_order[1]
        inner = boundaries==boundary_order[0]
        bbox = bound_props[boundary_order[1]].bbox
    return(inner,outer,bbox)


def hough_ellipse_finder():
    """
    Probably wont need
    :return:
    """
    result = hough_ellipse(edges, accuracy=20, threshold=250,
                           min_size=100, max_size=120)
    result.sort(order='accumulator')
    for ii in range(1,3):
        best = list(result[-ii])
        yc, xc, a, b = [int(round(x)) for x in best[1:5]]
        orientation = best[5]

        # Draw the ellipse on the original image
        cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
        # Draw the edge (white) and the resulting ellipse (red)
        edges = color.gray2rgb(img_as_ubyte(edges))
        edges[cy, cx] = (250, 0, 0)

def expand_bbox(box,expansion=0.4):
    """
    accepts a solid rectangle ROI and expands it
    :param box:
    :param expansion:
    :return:
    """
    x_bds = np.array([np.min(box[1]),np.max(box[1])])
    y_bds = np.array([np.min(box[0]),np.max(box[0])])
    w = np.abs(np.diff(x_bds))
    h = np.abs(np.diff(y_bds))
    pad_w = int(w*expansion/2)
    pad_h = int(h*expansion/2)
    x_bds+=[-pad_w,pad_w]
    y_bds+=[-pad_h,pad_h]
    return(draw.rectangle((y_bds[0], x_bds[0]), (y_bds[1], x_bds[1])))




def find_all_in_slice(I,fol_dict,slice_num):


    for fol in fol_dict.iteritems():

        rr,cc = expand_bbox(fol.bbox[slice_num])
        I_sub = I[rr,cc]
        inner,outer,bbox=extract_mask(I_sub)
        bbox = draw.rectangle(bbox[:2],bbox[2:])
        I_temp = color.gray2rgb(I_sub)
        bbox = expand_bbox(bbox)
        I_temp[inner] = (250,0,0)
        I_temp[outer] = (0,250,0)
        plt.imshow(I_temp)
        plt.title('Follicle {}'.format(id))
        plt.pause(0.2)
        inner_xpts,inner_ypts = np.where(inner)
        inner_xpts+=np.min(cc)
        inner_ypts+=np.min(rr)
        inner_pts = (inner_ypts,inner_xpts)
        outer_xpts,outer_ypts = np.where(outer)
        outer_xpts+=np.min(cc)
        outer_ypts+=np.min(rr)
        outer_pts = (outer_ypts,outer_xpts)

        fol.add_inner(slice_num,inner_pts)
        fol.add_outer(slice_num,inner_pts)
        fol.add_bbox(slice_num,bbox)













def propgate_ROI(I0,I1,fol_dict):
    """
    Use the found ellipses in the current stack to find the ROIs in the next stack.
    :param I0: The current image
    :param I1: The next image
    :param fol_dict: The dictionary of ROIs.
    :return: a new dictionary of ROIs.
    """
    pass


if __name__=='__main__':
    filename = r'L:\Users\guru\Documents\hartmann_lab\data\Pad2_2018\Pad2_2018\registered\regPad2_2018_0131.tif'
    I = io.imread(filename)
    major_box = ui_major_bounding_box(I)
    fol_dict = user_get_fol_bounds(I,major_box)
    rr = fol_dict[1][0]
    cc = fol_dict[1][1]
    I_sub = I[rr,cc]




