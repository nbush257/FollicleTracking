
from skimage import transform,draw,feature,io,filters,segmentation,color,exposure,morphology
from skimage.draw import ellipse_perimeter
from matplotlib import patches
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import closing,square,disk,label
from skimage.measure import regionprops,EllipseModel


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
        if slice_num in self.bbox.keys():
            print('Overwriting previous tracking on slice {}'.format(slice_num))
        else:
            self.bbox[slice_num] = box


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






def extract_mask(I_sub,thresh_size=1000):
    """
    Find ellipses in the follicle ROI
    :param I_sub: Sub image of just the follicle ROI
    :param size_thresh: number of pixels the follicle needs to be in order to count as a follicle candidate
    :return: inner, outer, bbox points of the inner and outer extents of the follicle
    """
    # Get the thresholded image to extract the follicle from
    fig,ax = plt.subplots(2,2)
    # I_sub = filters.median(I_sub)
    # I_sub = exposure.equalize_hist(I_sub)
    g = filters.frangi(I_sub)
    thresh_size = min(I_sub.shape)/5
    if thresh_size %2 ==0:
        thresh_size+=1
    T = filters.threshold_local(g,thresh_size)

    bw = g>T
    bw = closing(bw,disk(3))
    ax[0].imshow(g)
    ax[1].imshow(bw)

    # extract the desired region
    region_labels = label(bw)
    props = regionprops(region_labels)
    # remove small regions
    idx = np.argsort([r.filled_area for r in props])[-4:]
    for ii in range(np.max(region_labels)):
        if ii not in idx:
            bw[region_labels==ii+1]=0
    region_labels = label(bw)
    props = regionprops(region_labels)

    candidate = ROI_cost(props,I_sub)

    # Remove the non propdesired regions from the label set
    mask = region_labels==np.array(candidate+1,dtype='int64')
    region_labels[np.logical_not(mask)]=0
    ax[2].imshow(region_labels)
    plt.show()
    plt.pause(0.5)
    inner,outer,bbox = extract_boundaries(region_labels)
    return(inner,outer,bbox)


def extract_boundaries(region_labels):
    """
    Get the inner and outer boundaries
    :param region_labels: A labelled image of just the follicle
    :return: inner,outer,bbox: pixels of the image that represent the boundaries of the follicle
    """
    boundary = segmentation.find_boundaries(region_labels,mode='inner')
    boundaries = label(boundary)
    bound_props = regionprops(boundaries)
    boundary_order = np.argsort([r.filled_area for r in bound_props])+1
    if len(boundary_order)==1:
        outer = boundaries==boundary_order[0]
        inner = outer
        bbox = bound_props[0].bbox
        print('Inner and outer follicle are indistinguishable')
    else:
        outer = boundaries==boundary_order[1]
        inner = boundaries==boundary_order[0]
    bboxes = np.array([list(x.bbox) for x in bound_props])
    x_bot = np.min(bboxes[:,0],axis=0)
    y_bot = np.min(bboxes[:,1],axis=0)
    x_top = np.max(bboxes[:,2],axis=0)
    y_top = np.max(bboxes[:,3],axis=0)
    bbox = (x_bot,y_bot,x_top,y_top)


    return(inner,outer,bbox)


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

def find_all_bounding_boxes(I,size_thresh=500,cost_thresh=100):

    print('Finding bouding boxes...')
    g = filters.frangi(I)
    T = filters.threshold_li(g)
    bw = g>T
    bw = closing(bw,disk(5))
    skel = morphology.skeletonize(bw)
    region_labels= label(skel)
    props = regionprops(region_labels)
    for ii in range(np.max(region_labels)):
        region = props[ii]
        if props[ii].convex_area<size_thresh:
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
    plt.imshow(I)
    ax = plt.gca()
    for ii,region in enumerate(props):
        box = region.bbox
        bounding_box_dict[ii]=box

        #Plot
        coord = box[:2][::-1]
        w = box[3]-box[1]
        h = box[2]-box[0]
        rect = patches.Rectangle(coord,w,h,linewidth=2,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
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
    pts[0][pts[0]>I.shape[0]] = I.shape[0]-1
    pts[1][pts[1]>I.shape[1]] = I.shape[1]-1


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
        rr,cc = expand_bbox(fol.bbox[slice_num],0.5)
        rr[rr>=I.shape[0]]=I.shape[0]-1
        cc[cc>=I.shape[1]]=I.shape[1]-1
        I_sub = I[rr,cc]
        plt.imshow(I_sub)
        plt.pause(0.4)
        try:
            inner,outer,bbox=extract_mask(I_sub)
        except:
            print('Failed on Follicle; Skipping. Try hough transform?')
            continue
            # hough_ellipse_finder(I_sub)

        fig,ax = plt.subplots(1,2)
        # convert the bounding box from a list of 4 to a mask
        bbox = draw.rectangle(bbox[:2],bbox[2:])
        # Show the user the found follicle bounds
        I_temp = color.gray2rgb(I_sub)
        bbox = expand_bbox(bbox)
        I_temp[inner] = (250,0,0)
        I_temp[outer] = (0,250,0)
        ax[0].imshow(I_temp)
        ax[1].imshow(I)
        rect = ppatches.Rectangle(coord,w,h,linewidth=2,edgecolor='r',facecolor='none')
        ax[1].add_patch(rect)
        plt.title('Follicle {}'.format(id))
        plt.pause(0.2)

        # Map the ROI points back to the full image
        inner_xpts,inner_ypts = np.where(inner)
        inner_xpts+=np.min(cc)
        inner_ypts+=np.min(rr)
        inner_pts = (inner_ypts,inner_xpts)
        outer_xpts,outer_ypts = np.where(outer)
        outer_xpts+=np.min(cc)
        outer_ypts+=np.min(rr)
        outer_pts = (outer_ypts,outer_xpts)

        fix_bounds(inner_pts,I_temp)
        fix_bounds(outer_pts,I_temp)





        # Plot the found follicle
        Ifull_temp[inner_pts] = (0,250,0)
        Ifull_temp[outer_pts] = (250,0,0)

        # add the data to the follicle object
        fol.add_inner(slice_num,inner_pts)
        fol.add_outer(slice_num,inner_pts)
        fol.add_bbox(slice_num,bbox)

    # Show the user the tracked pad
    #TODO: Fix the mapping back to the major image
    plt.imshow(Ifull_temp)
    plt.pause(0)


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
    filename = r'C:\Users\nbush257\Desktop\regPad2_2018_0131.tif'
    I = io.imread(filename)
    major_box = ui_major_bounding_box(I)
    # fol_dict = user_get_fol_bounds(I,major_box,1)
    bbox_dict = find_all_bounding_boxes(I[major_box],size_thresh=700,cost_thresh=250)
    fol_dict = bbox_to_fol_dict(bbox_dict,1)
    find_all_in_slice(I[major_box],fol_dict,1)





