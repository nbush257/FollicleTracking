from scipy.spatial import distance
from skimage import transform,draw,feature,io,filters,segmentation,color,exposure,morphology,measure
from skimage.draw import ellipse_perimeter
from matplotlib import patches
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import closing,square,disk,label
from skimage.measure import regionprops,EllipseModel
import cPickle as pickle
import os
import sys


class Follicle():
    """
    An object which contains the points and metadata for a given follicle. Keeps inner, outer
    and bounding box information in dictionaries where the key
    is the slice number.
    """
    def __init__(self,ID):
        self.ID = ID
        self.whisker=''
        self.inner = defaultdict()
        self.outer = defaultdict()
        self.bbox = defaultdict()
        self.centroid = defaultdict()

    def add_inner(self,slice_num,inner_pts):
        """
        Add slice, points pair to the inner follicle dictionary
        :param slice_num: The identifying number for the slice
        :param inner_pts: The pixels defining the inner follicle
        :return: None
        """
        if slice_num in self.inner.keys():
            print('Overwriting previous tracking on slice {}'.format(slice_num))
        else:
            self.inner[slice_num] = inner_pts


    def add_outer(self,slice_num,outer_pts):
        """
        Add slice, points pair to the outer follicle dictionary
        :param slice_num: The identifying number for the slice
        :param outer_pts: The pixels defining the outer follicle
        :return: None
        """
        if slice_num in self.outer.keys():
            print('Overwriting previous tracking on slice {}'.format(slice_num))
        else:
            self.outer[slice_num] = outer_pts


    def add_bbox(self,slice_num,box):
        """
        Add slice, bounding box pair to the bounding box follicle dictionary
        :param slice_num: The identifying number for the slice
        :param box: The pixels defining the extent of the follicle in terms of a bounding box.
        :return: None
        """
        self.bbox[slice_num] = box

    def add_centroid(self, slice_num,centroid):
        self.centroid[slice_num] = centroid

def ginput2boundingbox(a,b,mode='mask'):
    """
    Takes the two corner points from the ginput of a rectangle
    and returns that rectangle in different forms
    :param a: (x_1,y_1)the first point
    :param b:(x_2,y_2) the second point
    :param mode: 'mask','verts',or 'patch' to define the output style
    :return: (rr,cc) if 'mask' or 'verts', (coord,h,w) if 'patch'
    """
    pts = np.array([[a[0],a[1]],[b[0],b[1]],[a[0],b[1]],[b[0],a[1]]],dtype='int')
    bot_x = np.min(pts[:,0])
    bot_y = np.min(pts[:,1])
    top_x = np.max(pts[:,0])
    top_y = np.max(pts[:,1])

    if mode=='mask':
        rr,cc = draw.rectangle((bot_x,bot_y),end=(top_x,top_y))
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

    return(cc,rr)


def ui_major_bounding_box(I):
    """
    Asks the user to define a major bounding box around the part of the pad
    that contains all the follicles.
    #TODO: Allow the user to refine the bounding box
    :param I: The image of the slice
    :return: (rr,cc) the rows and columns which define the kept parts of the image
    """
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
    """
    Given a bounding box in 'mask' form, zooms the passed axis to those points.
    Allows for viewing of the ROI without modifying the image
    :param ax: Axes which contain the image to be zoomed
    :param box: The bounding box in 'mask' form
    :return: None
    """
    # TODO: Allow box to be passed as a variety of input types
    ax.set_xlim(np.min(box[1]),np.max(box[1]))
    ax.set_ylim(np.min(box[0]),np.max(box[0]))
    ax.invert_yaxis()


def user_get_fol_bounds(I,major_box,slice_num,fol_dict=None):
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
        F.add_bbox(slice_num,box)
        fol_dict[count] = F
    plt.close('all')
    return(fol_dict)


def plot_bbox_from_region(region,ax=None):
    """
    If given a 'regionprops' region, plot a bounding box around that for feedback
    :param region: A regionprops region
    :param ax: the axes to plot the rectangle on
    :return: None
    """
    if ax is None:
        ax = plt.gca()
    bbox = region.bbox
    coord = bbox[1::-1]
    h = bbox[2] - bbox[0]
    w = bbox[3] - bbox[1]
    rect = patches.Rectangle(coord, w, h, linewidth=5, edgecolor='r', facecolor='none')
    ax.add_patch(rect)


def ROI_cost(props,I_sub):
    """
    Uses a combination of distance to center, size, and ellips-likeness
    to pick the best mask in the image
    :param props:
    :param I_sub:
    :return:
    """
    e_fit_cost = []
    center_dist_cost = []
    for region in props:
        e_fit_cost.append(get_ellipse_fit_cost(region))
    ellip_idx = np.argsort(e_fit_cost)
    size_idx = np.argsort([r.filled_area for r in props])[::-1]
    dist = lambda(r): np.sum((r.centroid - np.divide(I_sub.shape, 2)) ** 2)
    center_idx = np.argsort([dist(r) for r in props])
    cost = np.empty(len(props),dtype='int')
    for idx in range(len(props)):
        cost[idx] = np.where(size_idx==idx)[0]+np.where(center_idx==idx)[0]+np.where(ellip_idx==idx)[0]
    return(np.argmin(cost))


def centroid_distance_cost(region):
    """
    get the variance of the
    :param region:
    :return:
    """
    skel = morphology.skeletonize(region.image)
    skel_region = regionprops(skel.astype('uint8'))[0]
    D = distance.cdist(np.array(skel_region.centroid)[:,np.newaxis].T,skel_region.coords).ravel()
    return(np.var(D))


def extract_mask(I_sub,thresh_size=1000,ratio_thresh=0.05,area_thresh=1000):
    """
    Find ellipses in the follicle ROI
    :param I_sub: Sub image of just the follicle ROI
    :param size_thresh: number of pixels the follicle needs to be in order to count as a follicle candidate
    :return: inner, outer, bbox points of the inner and outer extents of the follicle
    """
    # Get the thresholded image to extract the follicle from
    # I_sub = filters.median(I_sub)
    I_sub = exposure.equalize_hist(I_sub)
    g = filters.frangi(I_sub)
    thresh_size = min(I_sub.shape)/5
    if thresh_size %2 ==0:
        thresh_size+=1
    T = filters.threshold_local(g,thresh_size)
    # T = filters.threshold_li(g)
    bw = g>T
    # fig,ax = plt.subplots(2,2)
    # ax[0,0].imshow(g)
    # ax[0,0].set_title('Frangi Filter')
    # ax[0,1].imshow(bw)
    # ax[0,1].set_title('All regions')

    # extract the desired region
    region_labels = label(bw)
    props = regionprops(region_labels)
    # remove small regions
    area = np.array([r.filled_area for r in props])
    area_ratio = np.array([np.divide(float(r.bbox_area),np.prod(I_sub.shape)) for r in props])
    idx = np.where(np.logical_and(area_ratio>ratio_thresh,area>area_thresh))[0]
    contains_centroid = [r.image[int(r.local_centroid[0]),int(r.local_centroid[1])] for r in props]
    for ii in range(np.max(region_labels)):
        if contains_centroid[ii]:# remove regions that are filled at the centroid
            bw[region_labels==ii+1]=0
        if ii not in idx: # remove regions that dont meet the threshold criterion.
            bw[region_labels==ii+1]=0

    region_labels = label(bw)
    props = regionprops(region_labels)
    candidate = ROI_cost(props,I_sub)
    # Chose the region chosen by ROI cost
    mask = region_labels==np.array(candidate+1,dtype='int64')
    remove_holes(mask)
    region_labels = mask.astype('int')

    props = regionprops(region_labels)
    # centroid = np.array(props[0].centroid)
    #
    # ax[1,0].imshow(region_labels)
    # ax[1,0].set_title('region chosen')
    #
    # region_labels = closing(region_labels,disk(7))
    # ax[1,1].imshow(region_labels)
    # ax[1,1].set_title('region closed')

    inner,outer,bbox,centroid = extract_boundaries(region_labels)
    return(inner,outer,bbox,centroid)


def extract_boundaries(region_labels):
    """
    Get the inner and outer boundaries
    :param region_labels: A labelled image of just the follicle
    :return: inner,outer,bbox: pixels of the image that represent the boundaries of the follicle
    """
    boundary = segmentation.find_boundaries(region_labels,mode='inner')
    boundaries = label(boundary)
    new_boundaries = np.zeros_like(boundaries)
    # try to remove funky regions
    for ii in range(np.max(boundaries)):
        edge = boundaries ==ii+1
        edge = morphology.skeletonize(closing(edge,disk(3)))
        new_boundaries[edge]=(ii+1)


    bound_props = regionprops(new_boundaries)
    boundary_order = np.argsort([r.convex_area for r in bound_props])+1
    if len(boundary_order)==1:
        outer = new_boundaries==boundary_order[0]
        inner = outer
        bbox = bound_props[0].bbox
        print('Inner and outer follicle are indistinguishable')
    else:
        outer = new_boundaries==boundary_order[1]
        inner = new_boundaries==boundary_order[0]
    # try to remove funky regions
    bboxes = np.array([list(x.bbox) for x in bound_props])
    x_bot = np.min(bboxes[:,0],axis=0)
    y_bot = np.min(bboxes[:,1],axis=0)
    x_top = np.max(bboxes[:,2],axis=0)
    y_top = np.max(bboxes[:,3],axis=0)
    bbox = (x_bot,y_bot,x_top,y_top)
    centroid = regionprops(label(inner))[0].centroid

    return(inner,outer,bbox,centroid)


def get_ellipse_fit_cost(region):
    pts = np.where(region.image)
    pts = np.array([pts[1],pts[0]]).T
    e = EllipseModel()
    e.estimate(pts)

    cost = np.mean(e.residuals(pts)**2)
    return(cost)


def hough_ellipse_finder(I_sub):
    """
    Uses the ellipse hough transform to find ellipses. Use if the bw theshold method fails.
    :param I_sub: The ROI image from which to extract the ellipses
    :return: innner,outer ellipse points.
    """
    edges = feature.canny(I_sub,sigma=2.0)
    result = transform.hough_ellipse(edges, accuracy=20, threshold=250,
                           min_size=100)
    result.sort(order='accumulator')

    for ii in range(len(result)):
        ellip = list(result[-ii])
        yc, xc, a, b = [int(round(x)) for x in ellip[1:5]]
        orientation = ellip[5]

        # Draw the ellipse on the original image
        cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
        cy = cy[cy<I_sub.shape[0]]
        cx = cx[cx<I_sub.shape[1]]
        # Draw the edge (white) and the resulting ellipse (red)
        Itemp = color.gray2rgb(I_sub)
        Itemp[cy, cx] = (250, 0, 0)
        plt.imshow(Itemp)
        plt.pause(1)


def find_all_bounding_boxes(I,min_size=500,max_size=1e6,cost_thresh=100):

    print('Finding bounding boxes...')
    g = filters.frangi(I)
    T = filters.threshold_li(g)
    bw = g>T
    bw = closing(bw,disk(5))
    skel = morphology.skeletonize(bw)
    region_labels= label(skel)
    props = regionprops(region_labels)
    for ii in range(np.max(region_labels)):
        region = props[ii]
        if (props[ii].convex_area<min_size) or (props[ii].bbox_area>max_size):
            skel[region_labels==(ii+1)] =0

    region_labels= label(skel)
    props = regionprops(region_labels)
    cost = np.array([get_ellipse_fit_cost(r) for r in props])

    for ii in range(np.max(region_labels)):
        region = props[ii]
        if cost[ii]>cost_thresh:
            skel[region_labels==(ii+1)] =0

    region_labels = label(skel)
    props = regionprops(region_labels)
    bounding_box_dict = defaultdict()
    if plot_tgl:
        plt.imshow(I)
        ax = plt.gca()
    for ii,region in enumerate(props):
        box = region.bbox
        bounding_box_dict[ii]=box

        #Plot
        if plot_tgl:
            coord = box[:2][::-1]
            w = box[3]-box[1]
            h = box[2]-box[0]
            rect = patches.Rectangle(coord,w,h,linewidth=2,edgecolor='r',facecolor='none')
            ax.add_patch(rect)

    if plot_tgl:
        plt.show()

    return(bounding_box_dict)


def bbox_to_fol_dict(bbox_dict,slice_num):
    fol_dict = defaultdict()
    count=0
    for box in bbox_dict.itervalues():
        count+=1
        F = Follicle(count)
        F.add_bbox(slice_num=slice_num,box=box)
        fol_dict[count] = F

    return(fol_dict)


def expand_bbox(box,expansion=0.5):
    """
    accepts a solid rectangle ROI and expands it
    :param box: A bounding box in "mask" form.
    :param expansion: the percentage to expand each dimension by [0,1]. Default: 0.5
    :return:
    """
    if len(box)==4:
        x_bds = np.array([box[1],box[3]])
        y_bds = np.array([box[0],box[2]])
    else:
        x_bds = np.array([np.min(box[1]),np.max(box[1])])
        y_bds = np.array([np.min(box[0]),np.max(box[0])])

    w = np.abs(np.diff(x_bds))
    h = np.abs(np.diff(y_bds))
    if type(expansion) is list:
        pad_w = expansion[0]
        pad_h = expansion[1]
    else:
        pad_w = int(w*expansion/2)
        pad_h = int(h*expansion/2)

    x_bds+=[-pad_w,pad_w]
    y_bds+=[-pad_h,pad_h]
    # check boundary of im
    x_bds[x_bds<0]=0
    y_bds[y_bds<0]=0

    return(draw.rectangle((y_bds[0], x_bds[0]), (y_bds[1], x_bds[1])))


def fix_bounds(pts,I):
    rr,cc = pts
    rr[rr>I.shape[0]] = I.shape[0]-1
    cc[cc>I.shape[1]] = I.shape[1]-1


def clean_region(region_bw,I_sub):
    vals = I_sub.ravel()
    bw_idx = region_bw.ravel().astype('bool')
    vals = vals[bw_idx]
    T = filters.threshold_yen(vals)
    new_bw = (I_sub<T)*region_bw
    # try find contours
    return(new_bw)


def remove_holes(region_bw):
    holes = label(np.invert(region_bw))
    props = regionprops(holes)
    idx = np.argsort([r.filled_area for r in props])
    corners =np.array([[0,0],
                       [0,region_bw.shape[1]-1],
                       [region_bw.shape[0]-1,0],
                       [region_bw.shape[0]-1,region_bw.shape[1]-1]])
    # This is probably not the prettiest implementation
    rm_idx = []
    for ii in idx:
        for corner in corners:
            if np.any(np.all(corner==props[ii].coords,axis=1)):
                rm_idx.append(ii)
    rm_idx = set(rm_idx)
    idx = [x for x in idx if x not in rm_idx]

    # Fill all but the last (largest) hole
    for ii in idx[:-1]:
        region_bw[props[ii].coords[:,0],props[ii].coords[:,1]] =True



def mask_final_points(I,pts,width=5):
    bw = np.zeros(I.shape,dtype='bool')
    bw[pts] = True
    bw = morphology.binary_dilation(bw,disk(4))
    return(np.where(bw)) #rr,cc

def extract_contour(edge,I_sub,mode):
    """
    use the found edges to define a mask and extract contours from that mask
    :param edge:
    :param I_sub:
    :param mode: 'inner' or 'outer'
    :return:
    """
    edge = morphology.dilation(edge,disk(5))
    vals = I_sub[np.where(edge)]
    T = filters.threshold_otsu(vals)
    if mode=='inner':
        contours = measure.find_contours(I_sub*edge,T)[0]
    elif mode == 'outer':
        contours = measure.find_contours(I_sub*edge,T,fully_connected='high')[1]
    return(contours)



def find_all_in_slice(I,fol_dict,slice_num):
    """
    Loop through all the follicle objects in fol dict and try to find
    the follicle in the image I.
    :param I: The image of the slice (grayscale)
    :param fol_dict: a dictionary of follicle objects. Uses the bounding box
                    for each follicle object to create the ROI on which to extract the follicle,
                    then adds information about the follicle
    :param slice_num: The slice identifying number. Used in the follicle object.
    :return: None
    """
    Ifull_temp = color.gray2rgb(I)

    for id,fol in fol_dict.iteritems():
        # Get an ROI and find the follicle
        plt.close('all')
        rr,cc = expand_bbox(fol.bbox[slice_num],0.5)
        rr[rr>=I.shape[0]]=I.shape[0]-1
        cc[cc>=I.shape[1]]=I.shape[1]-1
        I_sub = I[rr,cc].T

        try:
            inner,outer,bbox,centroid=extract_mask(I_sub)
        except:
            print('No data found in fol {}'.format(id))
            continue

        # convert the bounding box from a list of 4 to a mask
        bbox = draw.rectangle(bbox[:2],bbox[2:])
        # Show the user the found follicle bounds
        bbox = expand_bbox(bbox)
        # Plot
        if plot_tgl:
            I_temp = color.gray2rgb(I_sub)
            fig,ax = plt.subplots(1,2)
            I_temp[inner] = (250,0,0)
            I_temp[outer] = (0,250,0)
            ax[0].imshow(I_temp)
            ax[1].imshow(I,'gray')
            coord = fol.bbox[slice_num][:2]
            w = fol.bbox[slice_num][3]-fol.bbox[slice_num][1]
            h = fol.bbox[slice_num][2]-fol.bbox[slice_num][0]
            rect = patches.Rectangle(coord[::-1],w,h,linewidth=2,edgecolor='r',facecolor='none')
            ax[1].add_patch(rect)
            ax[0].set_title('Follicle {}'.format(id))
            ax[0].plot(centroid[1],centroid[0],'o')
            plt.show()
            plt.pause(0.1)

        # Map the ROI points back to the full image
        inner_row,inner_col = np.where(inner)
        inner_row+=np.min(rr)
        inner_col+=np.min(cc)
        inner_pts = (inner_row,inner_col)

        outer_row,outer_col = np.where(outer)
        outer_row+=np.min(rr)
        outer_col+=np.min(cc)
        outer_pts = (outer_row,outer_col)

        centroid += np.array([np.min(rr),np.min(cc)])
        fix_bounds(inner_pts,Ifull_temp)
        fix_bounds(outer_pts,Ifull_temp)
        mask_inner = mask_final_points(I,inner_pts)
        mask_outer = mask_final_points(I,outer_pts)

        # Plot the found follicle in full image
        Ifull_temp[mask_inner] = (0,250,0)
        Ifull_temp[mask_outer] = (250,0,0)

        # add the data to the follicle object
        fol.add_inner(slice_num,inner_pts)
        fol.add_outer(slice_num,inner_pts)
        fol.add_bbox(slice_num,bbox)
        fol.add_centroid(slice_num,centroid)# in row,col notation

    # Show the user the tracked pad
    plt.imshow(Ifull_temp)



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
    plot_tgl = True
    filename = r'C:\Users\nbush257\Desktop\regPad2_2018_0131.tif'
    filename = r'L:\Users\guru\Documents\hartmann_lab\data\Pad2_2018\Pad2_2018\registered\regPad2_2018_0131.tif'
    slice_num = int(filename[-8:-4])
    bbox_fname = r'C:\Users\guru\Desktop\bbox_{:04}.pckl'.format(slice_num)
    # bbox_fname = r'C:\Users\nbush257\Desktop\bbox_{:04}.pckl'.format(slice_num)
    I = io.imread(filename)
    # major_box = ui_major_bounding_box(I)
    # fol_dict = user_get_fol_bounds(I,major_box,1)

    # only have to get the bbox_dict once
    if os.path.isfile(bbox_fname):
        with open(bbox_fname,'r') as bbox_file:
            bbox_dict = pickle.load(bbox_file)
    else:
        bbox_dict = find_all_bounding_boxes(I,min_size=4000,cost_thresh=250)
        with open(bbox_fname,'w') as bbox_file:
            pickle.dump(bbox_dict,bbox_file)


    fol_dict = bbox_to_fol_dict(bbox_dict,slice_num)
    find_all_in_slice(I,fol_dict,slice_num)





