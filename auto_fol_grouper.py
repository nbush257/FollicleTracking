# TODO ESSENTIAL
# TODO handle slices with less than two matching pairs
# TODO make executable

# TODO WOULD BE NICE
# TODO make sure when relabeling left, right also gets relabeled
# TODO make rotation between +- pi
# TODO flesh out comments, maybe clean up code a little

from fol_IO import *
from helpers import *
from scipy.spatial.distance import *
import copy
import h5py


POINT_SIZE = 10
LABEL_THRESH = 100
SAME_FOL_DIST = 40
IDLE = 0
INIT = 1
MATCH_CENTROIDS = 2
LABEL = 3

# generate allowed labels 0:7, a:e
LABELS = []
for a in 'abcde':
    for i in range(1,8):
        LABELS.append(a + str(i))
for a in 'abcde':
        LABELS.append(a + str(0))
LABELS.append('xx')

# generate markers
MARKERS = []
for b in ['o', '^', 's', '+', 'd', '*']:
    for a in ['y', 'm', 'r', 'c', 'g', 'b', 'k']:
        MARKERS.append(a+b)
MARKERS.pop(-1)
MARKERS[-1] = 'w.'

# map markers to labels
P_TYPE = {}
for i in range(len(LABELS)):
    P_TYPE[LABELS[i]] = MARKERS[i]


class FolClicker(object):
    """
    graphic interface for aligning and labeling follicles
    """
    def __init__(self, slice_dict, img_dir=None):
        """
        process begins with object init, new window will open
        :param slice_dict:
        :param img_dir: the directory where slice images are stored images from only 1 pad should be in dir
        """
        # set image directory
        self.img_dir = img_dir
        if not self.img_dir:
            self.img_dir = os.getcwd()

        # create fig with 2 subplots
        self.fig = plt.figure(figsize=(14, 7))
        self.ax_left = self.fig.add_subplot(121)
        self.ax_right = self.fig.add_subplot(122)

        # attach mouse click and keypress events to fig
        self.cidb = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cidk = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # clean up slice dictionayr and add level for aligned/unaligned follicle data
        self.slice_dict = slice_dict
        self.slice_dict = remove_bbox_centroids(self.slice_dict)
        self.slice_dict = reshape_dict(self.slice_dict)

        # variable init
        self.click_state = IDLE
        self.key_state = 0
        self.curr_label = ''
        self.centroids_to_align = [[], []]
        self.fol_to_label = None
        self.slice_to_label = None
        self.ordered_slice_keys = sorted(slice_dict)
        self.ax_to_label = None

        # get index of starting slice
        self.start_idx = int(len(self.ordered_slice_keys)) / 2
        self.slice_key_idx = [self.start_idx, self.start_idx+1]
        self.curr_slice_key = [self.ordered_slice_keys[self.slice_key_idx[0]],
                               self.ordered_slice_keys[self.slice_key_idx[1]]]

        # show first slice
        self.set_key_state(INIT)
        self.ax_left.imshow(get_img_file(self.img_dir, self.curr_slice_key[0])[1], 'gray')
        self.flush()

    def on_click(self, event):
        """
        mouseclick event handler
        double click to label points
        single click selects centroids when in that state
        TODO should have two different states for single/double click
        :param event:
        :return:
        """
        if event.dblclick:
            # sets slice and follicle being labeled, sets key_input state to label
            if event.inaxes == self.ax_left:
                self.slice_to_label = self.curr_slice_key[0]
                self.ax_to_label = self.ax_left
                c, fk = find_closest_centroid([event.ydata, event.xdata],
                                              self.slice_dict[self.curr_slice_key[0]]['fols'])
            elif event.inaxes == self.ax_right:
                self.ax_to_label = self.ax_right
                self.slice_to_label = self.curr_slice_key[1]
                c, fk = find_closest_centroid([event.ydata, event.xdata],
                                              self.slice_dict[self.curr_slice_key[1]]['fols'])
            self.fol_to_label = fk
            self.set_key_state(LABEL)

        else:   # sigle click
            if self.click_state == MATCH_CENTROIDS:
                # when two points in each axis have been clicked, invoke procedure to find alignment, propegate labels,
                #   plots all centroids and labels
                if event.inaxes == self.ax_left:
                    if len(self.centroids_to_align[0]) < 2:
                        c, fk = find_closest_centroid([event.ydata, event.xdata],
                                                   self.slice_dict[self.curr_slice_key[0]]['fols'])
                        self.centroids_to_align[0].append(fk)
                        if len(self.centroids_to_align[0]) == 1:
                            color = 'r.'
                        else:
                            color = 'b.'
                        plot_point(self.ax_left, c, color)
                        self.fig.canvas.draw()
                        self.fig.canvas.flush_events()
                elif event.inaxes == self.ax_right:
                    if len(self.centroids_to_align[1]) < 2:
                        c, fk = find_closest_centroid([event.ydata, event.xdata],
                                                      self.slice_dict[self.curr_slice_key[1]]['fols'])
                        self.centroids_to_align[1].append(fk)
                        if len(self.centroids_to_align[1]) == 1:
                            color = 'r.'
                        else:
                            color = 'b.'
                        plot_point(self.ax_right, c, color)
                        self.fig.canvas.draw()
                        self.fig.canvas.flush_events()

                if len(self.centroids_to_align[0]) >= 2 and len(self.centroids_to_align[1]) >= 2:
                    points = [[],[]]
                    for i in range(2):
                        for k in self.centroids_to_align[i]:
                            points[i].append(self.slice_dict[self.curr_slice_key[i]]['fols'][k]['centroid'])
                    trans, rot, mid = align_slices(points)

                    self.slice_dict[self.curr_slice_key[1]]['rot_rad'] = \
                        rot + self.slice_dict[self.curr_slice_key[0]]['rot_rad']
                    self.slice_dict[self.curr_slice_key[1]]['trans'] = \
                        [trans[0] + self.slice_dict[self.curr_slice_key[0]]['trans'][0],
                         trans[1] + self.slice_dict[self.curr_slice_key[0]]['trans'][1]]
                    self.slice_dict[self.curr_slice_key[1]]['mid'] = mid
                    d = get_aligned_centroids(self.slice_dict[self.curr_slice_key[1]],
                                              self.slice_dict[self.curr_slice_key[1]]['trans'],
                                              self.slice_dict[self.curr_slice_key[1]]['rot_rad'],
                                              self.slice_dict[self.curr_slice_key[1]]['mid'])
                    self.slice_dict[self.curr_slice_key[1]]['aligned_fols'] = d

                    plot_centroids(self.ax_left, self.slice_dict[self.curr_slice_key[0]]['fols'], 'w.')
                    plot_centroids(self.ax_right, self.slice_dict[self.curr_slice_key[1]]['fols'], 'w.')
                    self.flush()
                    self.set_click_state(IDLE)
                    self.propagate_labels()

                    # self.change_click_state()

    def on_key(self, event):
        """
        key press event handler
        various actions based on state
        :param event:
        :return:
        """

        if self.key_state == LABEL:
            # expecting incoming label once two keys have been pressed, checks to make sure valid label, labels
            #   selected point and points within radius, propagates labels from left to right
            self.curr_label += event.key
            self.put_text(self.curr_label)
            if len(self.curr_label) >= 2:
                if self.curr_label in LABELS:
                    self.slice_dict[self.slice_to_label]['fols'][self.fol_to_label]['label'] = self.curr_label

                    # label follicles on top of each other the same
                    c = get_centroid_list(self.slice_dict[self.slice_to_label]['fols'])
                    sorted_keys_l = sorted(self.slice_dict[self.slice_to_label]['fols'].keys())
                    cd = cdist([self.slice_dict[self.slice_to_label]['fols'][self.fol_to_label]['centroid']], c)
                    idx = np.where((cd < SAME_FOL_DIST) & (cd != 0))
                    for n in range(len(idx[1])):
                        k = sorted_keys_l[int(idx[1][n])]
                        self.slice_dict[self.slice_to_label]['fols'][k]['label'] = self.curr_label

                    plot_point(self.ax_to_label,
                               self.slice_dict[self.slice_to_label]['fols'][self.fol_to_label]['centroid'],
                               P_TYPE[self.curr_label])
                    if self.ax_to_label is self.ax_left:
                        self.propagate_labels()
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                    self.curr_label = ''
                    self.set_key_state(IDLE)
                else:
                    self.curr_label = ''
                    # TODO instructions
        elif self.key_state == IDLE:
            if event.key == ' ':    # next slide
                if not self.next_slide():
                    self.fig.clear()
                    self.fig.suptitle('Writing Data')
                    self.flush()
                    self.write_h5()
                    self.fig.suptitle('Write Complete')
                    self.flush()
            elif event.key == 'r':  # reset centroid matching
                # clear labels on right
                for d in self.slice_dict[self.curr_slice_key[1]]['fols'].itervalues():
                    if 'label' in d.keys():
                        del d['label']
                self.ax_left.cla()
                self.ax_right.cla()

                self.ax_left.imshow(get_img_file(self.img_dir, self.curr_slice_key[0])[1], 'gray')
                self.ax_right.imshow(get_img_file(self.img_dir, self.curr_slice_key[1])[1], 'gray')

                plot_centroids(self.ax_left, self.slice_dict[self.curr_slice_key[0]]['fols'], 'g.')
                plot_centroids(self.ax_right, self.slice_dict[self.curr_slice_key[1]]['fols'], 'g.')

                self.set_click_state(MATCH_CENTROIDS)

            elif event.key == 'n':
                # TODO remove slice from dict
                pass
        elif self.key_state == INIT:
            # scroll trhough images to select image you would like to label first
            if event.key == 'left':
                self.start_idx -= 1
                self.ax_left.cla()
                self.slice_key_idx = [self.start_idx, self.start_idx + 1]
                self.curr_slice_key = [self.ordered_slice_keys[self.slice_key_idx[0]],
                                       self.ordered_slice_keys[self.slice_key_idx[1]]]
                self.ax_left.imshow(get_img_file(self.img_dir, self.curr_slice_key[0])[1], 'gray')
                self.flush()
            elif event.key == 'right':
                self.start_idx += 1
                self.ax_left.cla()
                self.slice_key_idx = [self.start_idx, self.start_idx + 1]
                self.curr_slice_key = [self.ordered_slice_keys[self.slice_key_idx[0]],
                                       self.ordered_slice_keys[self.slice_key_idx[1]]]
                self.ax_left.imshow(get_img_file(self.img_dir, self.curr_slice_key[0])[1], 'gray')
                self.flush()
            elif event.key == ' ':
                self.slice_dict[self.curr_slice_key[0]]['rot_rad'] = 0
                self.slice_dict[self.curr_slice_key[0]]['trans'] = [0, 0]
                self.slice_dict[self.curr_slice_key[0]]['mid'] = [0, 0]
                self.slice_dict[self.curr_slice_key[0]]['aligned_fols'] = \
                    copy.deepcopy(self.slice_dict[self.curr_slice_key[0]]['fols'])
                self.ax_left.imshow(get_img_file(self.img_dir, self.curr_slice_key[0])[1], 'gray')
                self.ax_right.imshow(get_img_file(self.img_dir, self.curr_slice_key[1])[1], 'gray')

                plot_centroids(self.ax_left, self.slice_dict[self.curr_slice_key[0]]['fols'], 'g.')
                plot_centroids(self.ax_right, self.slice_dict[self.curr_slice_key[1]]['fols'], 'g.')

                self.set_click_state(MATCH_CENTROIDS)
                self.set_key_state(IDLE)


    def put_text(self, text):
        """
        puts instructions on figure
        :param text:
        :return:
        """
        self.fig.suptitle(text)
        self.flush()

    def set_click_state(self, state):
        """
        click state switching actions
        :param state:
        :return:
        """
        if state == MATCH_CENTROIDS:
            self.centroids_to_align = [[], []]
            self.put_text('Click 2 pairs of matching centroids. Opposite corners are best.')

        self.click_state = state

    def set_key_state(self, state):
        """
        key state switching
        :param state:
        :return:
        """
        if state == LABEL:
            self.put_text('Type 2 digit label.')
        elif state == INIT:
            self.fig.suptitle('Use left/right arrow keys to select starting follicle. Press SPACE to continue.')
            self.flush()

        self.key_state = state

    def next_slide(self):
        """
        go to next slice, update values, update figure
        :return:
        """
        # change pictures
        # click centroid state
        # put up blank centroids
        # updatde current slide key
        # if idx is greater or less than to determine next

        for d in self.slice_dict[self.curr_slice_key[0]]['fols'].itervalues():
            if 'label' not in d.keys():
                d['label'] = 'xx'

        for d in self.slice_dict[self.curr_slice_key[1]]['fols'].itervalues():
            if 'label' not in d.keys():
                d['label'] = 'xx'

        self.set_click_state(MATCH_CENTROIDS)

        if self.slice_key_idx[1] >= len(self.ordered_slice_keys)-1:
            self.slice_key_idx = [self.start_idx, self.start_idx - 1]
        elif self.slice_key_idx[1] <= 0:
            return False
        elif self.slice_key_idx[0] < self.slice_key_idx[1]:
            self.slice_key_idx[0] += 1
            self.slice_key_idx[1] += 1
        elif self.slice_key_idx[0] > self.slice_key_idx[1]:
            self.slice_key_idx[0] -= 1
            self.slice_key_idx[1] -= 1

        self.ax_left.cla()
        self.ax_right.cla()

        self.curr_slice_key = [self.ordered_slice_keys[self.slice_key_idx[0]], self.ordered_slice_keys[self.slice_key_idx[1]]]

        self.ax_left.imshow(get_img_file(self.img_dir, self.curr_slice_key[0])[1], 'gray')
        self.ax_right.imshow(get_img_file(self.img_dir, self.curr_slice_key[1])[1], 'gray')

        plot_centroids(self.ax_left, self.slice_dict[self.curr_slice_key[0]]['fols'], 'g.')
        plot_centroids(self.ax_right, self.slice_dict[self.curr_slice_key[1]]['fols'], 'g.')

        self.set_click_state(MATCH_CENTROIDS)

        return True

    def propagate_labels(self):
        """
        attempts to automatically label follicles in right image based on labels from previous 3 slices
        :return:
        """
        # find closest LABELED centroid in left image to all centroids in right
        # if within thresh distance, adopt label
        # for all unlabeled points, check previous two slices

        sorted_keys_l = sorted(self.slice_dict[self.curr_slice_key[0]]['aligned_fols'].keys())
        sorted_keys_r = sorted(self.slice_dict[self.curr_slice_key[1]]['aligned_fols'].keys())

        for kl in sorted_keys_l:
            if 'label' in self.slice_dict[self.curr_slice_key[0]]['fols'][kl].keys():
                point = self.slice_dict[self.curr_slice_key[0]]['fols'][kl]['centroid']
                color = P_TYPE[self.slice_dict[self.curr_slice_key[0]]['fols'][kl]['label']]
                plot_point(self.ax_left, point, color)

        cr = get_centroid_list(self.slice_dict[self.curr_slice_key[1]]['aligned_fols'])
        for j in range(3):

            if self.slice_key_idx[0] < self.slice_key_idx[1]:   # going toward end of stack
                k_idx = self.slice_key_idx[0] - j
            else:
                k_idx = self.slice_key_idx[0] + j
            if k_idx >= len(self.ordered_slice_keys) or k_idx < 0:
                break
            slice_key = self.ordered_slice_keys[k_idx]
            if 'aligned_fols' not in self.slice_dict[slice_key]:
                break
            sorted_keys_l = sorted(self.slice_dict[slice_key]['aligned_fols'].keys())
            cl = get_centroid_list(self.slice_dict[slice_key]['aligned_fols'])
            cd = cdist(cl, cr)
            min_cols = cd.min(axis=0)  # distance to closest point in first slice to all points in second slice
            idx = []

            for m in min_cols:
                idx.append(np.where(cd == m))

            for n in range(len(min_cols)):
                if min_cols[n] < LABEL_THRESH:
                    if len(idx[n][0]) > 1:
                        m = np.where(idx[n][1] == n)
                        kl = sorted_keys_l[int(idx[n][0][m])]
                    else:
                        kl = sorted_keys_l[int(idx[n][0])]
                    kr = sorted_keys_r[n]
                    if 'label' in self.slice_dict[slice_key]['fols'][kl].keys():
                        if self.slice_dict[slice_key]['fols'][kl]['label'] != 'xx':
                            if 'label' not in self.slice_dict[self.curr_slice_key[1]]['fols'][kr].keys():
                                self.slice_dict[self.curr_slice_key[1]]['fols'][kr]['label'] = \
                                    self.slice_dict[slice_key]['fols'][kl]['label']

        # Plot
        for kr in sorted_keys_r:
            if 'label' in self.slice_dict[self.curr_slice_key[1]]['fols'][kr].keys():
                point = self.slice_dict[self.curr_slice_key[1]]['fols'][kr]['centroid']
                color = P_TYPE[self.slice_dict[self.curr_slice_key[1]]['fols'][kr]['label']]
                plot_point(self.ax_right, point, color)

        self.flush()

    def flush(self):
        """
        update figure
        :return:
        """
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def write_h5(self, **kwargs):
        """
        fills in aligned_fols in slice dict
        writes to hdf5 file
        :param kwargs:
        :return:
        """
        self.get_aligned_fol()
        self.get_aligned_bbox()
        self.get_labels()

        if 'title' in kwargs.keys():
            title = kwargs['title']
        else:
            title = self.img_dir.split('\\')[-1] + '_labeled.h5'

        if 'destination_folder' in kwargs.keys():
            destination_folder = kwargs['destination_folder']
        else:
            destination_folder = self.img_dir

        file_str = os.path.join(destination_folder, title)

        with h5py.File(file_str, 'w') as pad:
            for ks, s in self.slice_dict.iteritems():
                slide = pad.create_group(str(ks))
                slide.attrs['unaligned_image_file'] = get_img_file(self.img_dir, ks)[0].split('\\')[-1]
                slide.attrs['rotation_radians'] = s['rot_rad']
                slide.attrs['rotation_origin'] = s['mid']
                slide.attrs['translation'] = s['trans']

                aligned = slide.create_group('aligned_folicles')
                unaligned = slide.create_group('unaligned_folicles')

                for kf, f in s['aligned_fols'].iteritems():
                    if f['label'] != 'xx':
                        fol = aligned.create_group(str(kf))
                        fol.attrs['label'] = f['label']
                        fol.attrs['centroid'] = f['centroid']
                        fol.attrs['bbox'] = f['bbox']

                        outer = fol.create_dataset('outer', shape=(len(f['outer'][0]), 2), dtype='float')
                        outer[:, 0] = f['outer'][0]
                        outer[:, 1] = f['outer'][1]

                        inner = fol.create_dataset('inner', shape=(len(f['inner'][0]), 2), dtype='float')
                        inner[:, 0] = f['inner'][0]
                        inner[:, 1] = f['inner'][1]

                for kf, f in s['fols'].iteritems():
                    if f['label'] != 'xx':
                        fol = unaligned.create_group(str(kf))
                        fol.attrs['label'] = f['label']
                        fol.attrs['centroid'] = f['centroid']
                        fol.attrs['bbox'] = f['bbox']

                        outer = fol.create_dataset('outer', shape=(len(f['outer'][0]), 2), dtype='float')
                        outer[:, 0] = f['outer'][0]
                        outer[:, 1] = f['outer'][1]

                        inner = fol.create_dataset('inner', shape=(len(f['inner'][0]), 2), dtype='float')
                        inner[:, 0] = f['inner'][0]
                        inner[:, 1] = f['inner'][1]

    def get_labels(self):
        """
        gets labels from fols and adds them to aligned_fols
        :return:
        """
        for s in self.slice_dict.itervalues():
            for k, f in s['fols'].iteritems():
                s['aligned_fols'][k]['label'] = f['label']

    def get_aligned_bbox(self):
        """
        gets bounding box from fol and rotate/translates to align, and adds to aligned_fols
        :return:
        """
        for s in self.slice_dict.itervalues():
            for k, f in s['fols'].iteritems():
                trans = s['trans']
                r = s['rot_rad']
                mid = s['mid']

                bbox = f['bbox']
                bbox = [[bbox[0], bbox[1]], [bbox[2], bbox[3]]]

                bbox = rotate_by_rad(bbox, mid, r)
                bbox = translate(bbox, trans)
                bbox = (bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])

                s['aligned_fols'][k]['bbox'] = bbox


    def get_aligned_fol(self):
        """
        gets outer and inner countour infromation from fols, rotates/translates, adds to aligned_fols
        :return:
        """
        for s in self.slice_dict.itervalues():
            if 'aligned_fols' not in s.keys():
                s['aligned_fols'] = {}
            for k, f in s['fols'].iteritems():
                trans = s['trans']
                r = s['rot_rad']
                mid = s['mid']
                # sorted_keys = sorted(s['fols'].keys())

                contour = get_contour(f, 'outer')
                rotated_contour = rotate_by_rad(contour, mid, r)
                translated_contour = translate(rotated_contour, trans)
                reshaped_contour = [[], []]
                for c in translated_contour:
                    reshaped_contour[0].append(c[0])
                    reshaped_contour[1].append(c[1])

                if k not in s['aligned_fols'].keys():
                    s['aligned_fols'][k] = {}
                s['aligned_fols'][k]['outer'] = reshaped_contour

                if 'inner' in f.keys():
                    contour = get_contour(f, 'inner')
                    if not contour:
                        s['aligned_fols'][k]['inner'] = [[], []]
                        break
                    rotated_contour = rotate_by_rad(contour, mid, r)
                    translated_contour = translate(rotated_contour, trans)
                    reshaped_contour = [[], []]
                    for c in translated_contour:
                        reshaped_contour[0].append(c[0])
                        reshaped_contour[1].append(c[1])

                    s['aligned_fols'][k]['inner'] = reshaped_contour


def remove_bbox_centroids(sd):
    """
    Remove 'follicles' that did not have contours tracked
    :param sd: full slice dictionary
    :return: the ammended slice dict. this is redundant TODO
    """
    to_del = []
    for sk in sd:
        for fk in sd[sk]:
            if 'outer' in sd[sk][fk].keys():
                if not (sd[sk][fk]['outer'] and sd[sk][fk]['centroid'].any()):
                    to_del.append((sk, fk))
            else:
                to_del.append((sk, fk))
    for d in to_del:
        del (sd[d[0]][d[1]])

    return sd


def get_centroid_list(d_slice):
    """
    ruturns ordered list of centroids from the follicles in passed in slice
    :param d_slice:
    :return: list of points [[x, y],[x, y]...]
    """
    sorted_keys = sorted(d_slice)

    centroids = []

    for k in sorted_keys:
        centroids.append(d_slice[k]['centroid'])

    return centroids


def find_closest_centroid(point, d_slice):
    """
    return the centroid and corresponding follicle index of centroid nearest to a given point from slice
    will break if two are equally close TODO
    :param point:
    :param d_slice:
    :return:
    """
    # find the key id of the closest follicle
    centroids = get_centroid_list(d_slice)
    cd = cdist([point], centroids)
    idx = np.argmin(cd)
    return centroids[idx], sorted(d_slice)[idx]  # key of centroid nearest to point


def plot_point(ax, point, color):
    """
    add a point to the axis
    x and y are inverted in image compared to stored data
    :param ax:
    :param point:
    :param color: color and marker type -eg 'r.'
    :return:
    """
    ax.plot(point[1], point[0], color)


def get_img_file(my_dir, key):
    """
    gives you the corresponding full filepath and image file for image matching the passed in slice index
    :param my_dir: local directory containing slice images
    :param key: index of desired slice
    :return: filename, image file
    """
    key = str(key)
    f_name = None
    while len(key) < 4:
        key = '0' + key
    key = key + '.tif'
    for f in os.listdir(my_dir):
        if key in f:
            f_name = f

    if f_name:
        return f_name, io.imread(os.path.join(my_dir, f_name))
    else:
        print('Image file not found')
        return None, None

# HERE Changing plot centroids to swap xy first

def plot_centroids(ax, d_slice, color):
    """
    plots centroids from slice on axis
    :param ax:
    :param d_slice:
    :param color: color and marker type -eg 'r.'
    :return:
    """
    for f in d_slice.itervalues():
        if type(f) is dict:
            plot_point(ax, f['centroid'], color)


def rotate_by_deg(points, origin, r):
    """
    rotate list of points by degrees about origin
    :param points: list of points [[x, y], [x, y]...]
    :param origin: point [x, y] to rotate all other points about
    :param r: angle to rotate in deg
    :return: list of rotated points
    """
    r = np.radians(r)
    rot = []
    cos_r, sin_r = np.cos(r), np.sin(r)
    ox, oy = origin[0], origin[1]
    for p in points:
        rot.append([ox + (p[0]-ox) * cos_r - (p[1]-oy) * sin_r, oy + (p[0]-ox) * sin_r + (p[1]-oy) * cos_r])
    return rot

def rotate_by_rad(points, origin, r):
    """
    rotate list of points by degrees about origin
    :param points: list of points [[x, y], [x, y]...]
    :param origin: point [x, y] to rotate all other points about
    :param r: angle to rotate in radians
    :return: list of rotated points
    """
    rot = []
    cos_r, sin_r = np.cos(r), np.sin(r)
    ox, oy = origin[0], origin[1]
    for p in points:
        rot.append([ox + (p[0]-ox) * cos_r - (p[1]-oy) * sin_r, oy + (p[0]-ox) * sin_r + (p[1]-oy) * cos_r])
    return rot


def my_dist(a, b):
    """
    return euclidian distance of two points
    :param a:
    :param b:
    :return:
    """
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** .5


def reshape_dict(slice_dict):
    """
    adds another layer in dict hierarchy to allow for separation of original and realigned points
    :param slice_dict:
    :return: ammended dict. this is redundant TODO
    """
    for k in slice_dict:
        to_del = slice_dict[k].keys()
        slice_dict[k]['fols'] = copy.deepcopy(slice_dict[k])
        for d in to_del:
            del(slice_dict[k][d])

    return slice_dict

def align_slices(points):
    """
    finds alignment for two slices based on 2 pairs of matching points
    lines up the midpoints of selected points from each slice, and finds rotation to minimize distace between
        matching points
    returns values to match second set slice to first
    :param points: [[[x00, y00],[x01, y01]], [[x10, y10],[x11, y11]]]
    :return: translation [x, y], rotation (radians), point about which second set was rotated
    """
    midpoint1 = [(points[0][0][0] + points[0][1][0])/2., (points[0][0][1] + points[0][1][1])/2]
    midpoint2 = [(points[1][0][0] + points[1][1][0]) / 2., (points[1][0][1] + points[1][1][1]) / 2]
    trans = [midpoint1[0] - midpoint2[0], midpoint1[1] - midpoint2[1]]

    error = []
    for r in range(360):
        rot = rotate_by_deg(points[1], midpoint2, r)
        mp = []
        for p in rot:
            mp.append([p[0] + trans[0], p[1] + trans[1]])
        error.append(my_dist(points[0][0], mp[0]) + my_dist(points[0][1], mp[1]))

    rad = error.index(min(error))

    error = []
    rotation = []
    for r in range(rad*10-9, rad*10 +10):
        rot = rotate_by_deg(points[1], midpoint2, r/10.)
        for p in rot:
            mp.append([p[0] + trans[0], p[1] + trans[1]])
        error.append(my_dist(points[0][0], mp[0]) + my_dist(points[0][1], mp[1]))
        rotation.append(r)

    rad = rotation[error.index(min(error))]/10.
    rad = np.radians(rad)

    return trans, rad, midpoint2


def translate(points, trans):
    """
    translates list of point by [x,y]
    :param points:
    :param trans:
    :return:
    """
    translated_points = []
    for p in points:
        translated_points.append([p[0] + trans[0], p[1] + trans[1]])
    return translated_points


def get_aligned_centroids(d_slice, trans, rot, mid):
    """
    rotate and translate all points in slice and add to slice['aligned_fols']
    this is nasty, should just ammend dict, and trans rot mid already in dict, don't need to pass
    :param d_slice:
    :param trans:
    :param rot:
    :param mid:
    :return:
    """
    sorted_keys = sorted(d_slice['fols'].keys())
    centroids = get_centroid_list(d_slice['fols'])
    rotated_centroids = rotate_by_rad(centroids, mid, rot)
    translated_centroids = translate(rotated_centroids, trans)

    if 'aligned_fols' in d_slice.keys():
        my_dict = d_slice['aligned_fols']
    else:
        my_dict = {}

    for i in range(len(sorted_keys)):
        k = sorted_keys[i]
        if k in my_dict.keys():
            my_dict[k]['centroid'] = translated_centroids[i]
        else:
            my_dict[k] = {}
            my_dict[k]['centroid'] = translated_centroids[i]

    return my_dict



def get_contour(fol, key):
    """
    converts contour in dict to list of points
    :param d_slice:
    :return: list of points [[x, y],[x, y]...]
    """
    contour = []
    if key in fol.keys():
        for n in range(len(fol[key][0])):
            contour.append([fol[key][0][n], fol[key][0][n]])
    return contour

if __name__ == '__main__':
    # arg - directory with images/slice dict
    # arg(optional) - title of h5 file
    # arg(optional) - dir to store h5 file
    # read in slice dict
    # start tracker
    pass



