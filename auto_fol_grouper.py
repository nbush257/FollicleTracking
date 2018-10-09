# TODO propagate labels, iter frames, key commands

from fol_IO import *
from helpers import *
from scipy.spatial.distance import *
import copy

POINT_SIZE = 10

IDLE = 0
INIT = 1
MATCH_CENTROIDS = 2
LABEL = 3

LABELS = []
for a in 'abcde':
    for i in range(1,7):
        LABELS.append(a + str(i))
for a in 'abcde':
        LABELS.append(a + str(0))

MARKERS = []

for b in ['o', '^', 's', '+', 'd', '*']:
    for a in ['b', 'g', 'r', 'c', 'm', 'y']:
        MARKERS.append(a+b)

P_TYPE = {}
for i in range(len(LABELS)):
    P_TYPE[LABELS[i]] = MARKERS[i]

def remove_bbox_centroids(sd):
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

    :param d_slice:
    :return:
    """
    sorted_keys = sorted(d_slice)

    centroids = []

    for k in sorted_keys:
        centroids.append(d_slice[k]['centroid'])

    return centroids


def find_closest_centroid(point, d_slice):
    # find the key id of the closest follicle
    centroids = get_centroid_list(d_slice)
    cd = cdist([point], centroids)
    idx = np.argmin(cd)
    return centroids[idx], sorted(d_slice)[idx]  # key of centroid nearest to point


def calc_rotation(centroids_to_allign):
    """
    calculate midpoints
    translate second midpoint to match first
    iterate over rotations to minimize distance between correspoinding centroids
    :param centroids_to_allign:
    :return: rotation in radians
    """
    return 0


def match_labels(sd1, sd2):
    """

    :param sd1:
    :param sd2:
    :return:
    """
    pass


def plot_point(ax, point, color):
    ax.plot(point[1], point[0], color)


def get_img_file(my_dir, key):
    key = str(key)
    f_name = None
    while len(key) < 4:
        key = '0' + key
    key = key + '.tif'
    for f in os.listdir(my_dir):
        if key in f:
            f_name = f

    if f_name:
        return io.imread(os.path.join(my_dir, f_name))
    else:
        return None

# HERE Changing plot centroids to swap xy first

def plot_centroids(ax, d_slice, color):
    for f in d_slice.itervalues():
        if type(f) is dict:
            plot_point(ax, f['centroid'], color)


def rotate_by_deg(points, origin, r):
    r = np.radians(r)
    rot = []
    cos_r, sin_r = np.cos(r), np.sin(r)
    ox, oy = origin[0], origin[1]
    for p in points:
        rot.append([ox + (p[0]-ox) * cos_r - (p[1]-oy) * sin_r, oy + (p[0]-ox) * sin_r + (p[1]-oy) * cos_r])
    return rot


def my_dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** .5


def reshape_dict(slice_dict):
    for k in slice_dict:
        to_del = slice_dict[k].keys()
        slice_dict[k]['fols'] = copy.deepcopy(slice_dict[k])
        for d in to_del:
            del(slice_dict[k][d])

    return slice_dict

def align_slices(points):
    midpoint1 = [(points[0][0][0] + points[0][1][0])/2., (points[0][0][1] + points[0][1][1])/2]
    midpoint2 = [(points[1][0][0] + points[1][1][0]) / 2., (points[1][0][1] + points[1][1][1]) / 2]
    trans = [midpoint1[0] - midpoint2[0], midpoint1[1] - midpoint2[1]]

    error = []
    for r in range(360):
        rot = rotate_by_deg(points[1], midpoint2, r)
        # add translation to rotated midpoints TODO
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

    return trans, rad


class FolClicker(object):
    def __init__(self, slice_dict, img_dir=None):
        self.img_dir = img_dir
        if not self.img_dir:
            self.img_dir = os.getcwd()
        self.fig = plt.figure(figsize=(14, 7))
        self.ax_left = self.fig.add_subplot(121)
        self.ax_right = self.fig.add_subplot(122)

        self.slice_dict = slice_dict
        self.slice_dict = remove_bbox_centroids(self.slice_dict)
        self.slice_dict = reshape_dict(self.slice_dict)

        self.cidb = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cidk = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.click_state = MATCH_CENTROIDS
        self.key_state = 0

        self.curr_label = ''

        self.centroids_to_align = [[], []]

        self.fol_to_label = None
        self.slice_to_label = None

        self.ordered_slice_keys = sorted(slice_dict)
        idx = int(len(self.ordered_slice_keys)) / 2
        self.curr_slice_key = [self.ordered_slice_keys[idx], self.ordered_slice_keys[idx + 1]]
        self.slice_dict[self.curr_slice_key[0]]['rot_rad'] = 0
        self.slice_dict[self.curr_slice_key[0]]['trans'] = [0, 0]

        self.ax_left.imshow(get_img_file(self.img_dir, self.curr_slice_key[0]), 'gray')
        self.ax_right.imshow(get_img_file(self.img_dir, self.curr_slice_key[1]), 'gray')

        plot_centroids(self.ax_left, self.slice_dict[self.curr_slice_key[0]]['fols'], 'g.')
        plot_centroids(self.ax_right, self.slice_dict[self.curr_slice_key[1]]['fols'], 'g.')

        self.ax_to_label = None

        # TODO rot/trans of first slice to 0

    def change_click_state(self, state):
        self.click_state = state

    def change_key_state(self, state):
        self.key_state = state

    def on_click(self, event):
        if event.dblclick:
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

            self.change_key_state(LABEL)

            # label point
            # if label in left, propagate to right
            pass
        else:
            if self.click_state == MATCH_CENTROIDS:
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
                    trans, rot = align_slices(points)

                    self.slice_dict[self.curr_slice_key[1]]['rot_rad'] = \
                        rot + self.slice_dict[self.curr_slice_key[0]]['rot_rad']
                    self.slice_dict[self.curr_slice_key[1]]['trans'] = \
                        [trans[0] + self.slice_dict[self.curr_slice_key[0]]['trans'][0],
                         trans[1] + self.slice_dict[self.curr_slice_key[0]]['trans'][1]]

                    plot_centroids(self.ax_left, self.slice_dict[self.curr_slice_key[0]]['fols'], 'w.')
                    plot_centroids(self.ax_right, self.slice_dict[self.curr_slice_key[1]]['fols'], 'w.')
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                    self.change_click_state(IDLE)
                    # TODO propagate labels
                    # self.change_click_state()



    def on_key(self, event):
        # keypress reader

        """
        r = Reset centroid matching
        space = next
        show rotated images
        plot follicles
        k - show label Key

        :param event:
        :return:
        """

        if self.key_state == LABEL:
            self.curr_label += event.key
            print(self.curr_label)
            if len(self.curr_label) >= 2:
                if self.curr_label in LABELS:
                    self.slice_dict[self.slice_to_label]['fols'][self.fol_to_label]['label'] = self.curr_label
                    plot_point(self.ax_to_label,
                               self.slice_dict[self.slice_to_label]['fols'][self.fol_to_label]['centroid'],
                               P_TYPE[self.curr_label])
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                    self.curr_label = ''
                    self.change_key_state(IDLE)
                    # TODO propagate labels
                    # TODO printout on fig?
                else:
                    self.curr_label = ''
                    # TODO printout on fig?
                    pass  # reset label

