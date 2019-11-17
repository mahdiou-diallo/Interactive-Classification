import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from matplotlib.backend_bases import MouseButton

class BinDataGenerator():
    """Class to create a dataset with two classes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes,

    clf : any classifier with a `fit()` and `predict()` methods

    labels : int,
        The labels for each class in the dataset
    
    markers : str or int,
        The marker used to represent each class on the plot
        Must be valid `matplotlib.markers` values

    colors : str,
        The colors used to represent each class on the plot
        Must be valid `matplotlib.colors` values 

    """
    def __init__(self, ax, clf, labels=[1, -1], markers=['o', 'x'], colors=['blue', 'red']):
        assert len(markers) == len(colors) == 2 # for now let's do two classes
        self.ax = ax
        self.labels = labels
        self.markers = markers
        self.colors = dict(zip(labels, colors))
        self.clf = clf

        self.nclass = len(colors)
        self.cur_class = 0

        self.data = []
        self.target = []
        self.finished = False

        self.ax.set_title(f'Left-Click to add points for class {self.labels[self.cur_class]}\nPress {NEXT_LABEL_BUTTON!r} for next class')
    

    def add_point(self, x, y):
        """Add a new point with the current label and plot it on the figure

        Parameters:
        -----------
        x : float
        y : float


        """
        if self.finished:
            return
        
        self.data.append([x, y])
        self.target.append(self.labels[self.cur_class])

        self.ax.scatter(x, y,
                     marker=self.markers[self.cur_class],
                     c=self.colors[self.labels[self.cur_class]])
        plt.draw()
    

    def next_class(self):
        self.cur_class += 1
        if self.cur_class >= self.nclass:
            self._finish()
            return
        
        self.ax.set_title(f'Left-Click to add points for class {self.labels[self.cur_class]}\nPress {NEXT_LABEL_BUTTON!r} for next class')
        plt.draw()
    
    
    def _finish(self):
        """Trains the classifier on the data and plots the decision boundary"""
        if self.finished:
            return

        self.finished = True
        X = np.array(self.data)
        y = np.array(self.target)
        self.clf.fit(X, y)

        xlims = self.ax.get_xlim()
        ylims = self.ax.get_ylim()
        xvals = np.arange(*xlims, step=(xlims[1]-xlims[0])/100)
        yvals = np.arange(*ylims, step=(ylims[1]-ylims[0])/100)

        xv, yv = np.meshgrid(xvals, yvals)
        grid = np.array([*zip(xv.reshape((-1,)), yv.reshape((-1,)))])
        pred = self.clf.predict(grid)

        cols = [self.colors[c] for c in pred]
        self.ax.scatter(grid[:, 0], grid[:, 1], marker='.', s=.5, c=cols)
        self.ax.set_title('Classifier decision boundary')

        plt.draw()

        # return {
        #     'ax': ax,
        #     'X': X,
        #     'y': y
        # }

ADD_POINT_KEY = MouseButton.LEFT
NEXT_LABEL_BUTTON = 'enter'

class EventHandler():
    def __init__(self, gen):
        self.gen = gen

    def onclick(self, event):
        # print('click', event.button)
        # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #     ('double' if event.dblclick else 'single', event.button,
        #     event.x, event.y, event.xdata, event.ydata))
        if event.button != ADD_POINT_KEY:
            return

        self.gen.add_point(event.xdata, event.ydata)

    def onpress(self, event):
        # print('press', event.key)
        if event.key != NEXT_LABEL_BUTTON:
            return
        self.gen.next_class()

if __name__ == '__main__':
    fig, ax = plt.subplots()

    ax.set_ylim(-10, 10)
    ax.set_xlim(-10, 10)

    # create dataset generator and event handler
    generator = BinDataGenerator(ax, GaussianNB())
    eh = EventHandler(generator)
    # attach event handlers
    btn_cid = fig.canvas.mpl_connect('button_press_event', eh.onclick)
    key_cid = fig.canvas.mpl_connect('key_press_event', eh.onpress)



    plt.show()

    # detach event handlers
    fig.canvas.mpl_disconnect(btn_cid)
    fig.canvas.mpl_disconnect(key_cid)
