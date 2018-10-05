from fol_IO import *
from helpers import *
from scipy.spatial.distance import *


class fol_clicker(object):
    def __init__(self):
        self.fig = plt.figure(figsize=(14, 7))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)


        cid = fig.canvas.mpl_connect('button_press_event', onclick)

    def onclick(event):
        #     global click_points
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))

        if event.inaxes == ax:
            print('ax')
        elif event.inaxes == ay:
            print('Y')

        if event.dblclick:
            click_points.append([event.xdata, event.ydata])

    cid = fig.canvas.mpl_connect('button_press_event', onclick)