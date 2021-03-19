"""
Code starting point taken from (17/03-2021):
https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot
"""
#import matplotlib
#matplotlib.use('nbAgg') # Tried to use same backend as jupyter notebook
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def projection_rejection(u,v):
    """
    projection of u on v, and rejection of u from v
    """
    proj = ((u@v)/(v@v)) * v
    reject = u - proj
    return proj, reject

class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""

    def __init__(self, cluster_data, weight_data, loss_history):
        """
        Args:
            cluster_data: (N,2), where N=4*n -> four clusters
            weight_data: (#epochs,2,4)
        """
        self.cluster_data = cluster_data
        self.weight_data = weight_data
        self.loss_history = loss_history

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.

        self.save_count=weight_data.shape[0]
        self.animation = animation.FuncAnimation(
            self.fig,
            self.update,
            init_func=self.setup_plot,
            interval=25, # time in ms between frames
            #repeat_delay=1000, # delay before loop
            blit=False, # for OSX?
            save_count=self.save_count, # #frames
        )

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        data = self.cluster_data
        colors = ["red", "green", "blue", "orange"]
        self.means = np.mean(
            np.reshape(data, (4, int(data.shape[0] / 4), data.shape[-1])), axis=1
        )
        color_idxs = np.argmax(self.means @ self.weight_data[0],axis=0) # shapes (4,2) @ (2,4)
        cluster_N = int(data.shape[0] / 4)
        self.weight_arrows = []
        self.rejection_arrows = []
        self.projection_arrows = []
        for color, mean, i in zip(colors, self.means, range(len(colors))):
            self.ax.scatter(
                data[i * cluster_N : (i + 1) * cluster_N, 0],
                data[i * cluster_N : (i + 1) * cluster_N, 1],
                color=color,
            )
            self.ax.arrow(
                0,
                0,
                *mean,
                length_includes_head=True,
                width=0.01,
                color=(0, 0, 0, 0.5),  # semi-transparent black arrow
            )
            
            warrow = self.ax.arrow(
                0,
                0,
                *self.weight_data[0, :, i],
                length_includes_head=True,
                width=0.01,
                color=colors[color_idxs[i]],#rscolor=(1, 0, 0, 0.5),  # semi-transparent green arrow
            )

            proj, reject = projection_rejection(self.weight_data[0, :, i],mean)
            rej_arrow = self.ax.arrow(
                *proj,
                *reject,
                length_includes_head=True,
                width=0.01,
                color=colors[color_idxs[i]],#rscolor=(1, 0, 0, 0.5),  # semi-transparent green arrow
                alpha=0.35,
            )

            proj_arrow = self.ax.arrow(
                0,
                0,
                *proj,
                length_includes_head=True,
                width=0.01,
                color=colors[color_idxs[i]],#rscolor=(1, 0, 0, 0.5),  # semi-transparent green arrow
                alpha=0.35,
            )
            self.weight_arrows.append(warrow)
            self.rejection_arrows.append(rej_arrow)
            self.projection_arrows.append(proj_arrow)
        self.ax.grid("on")
        self.ax.set_title('Loss={}'.format(self.loss_history[0]))

    def update(self, k):
        """Update the scatter plot."""
        colors = ["red", "green", "blue", "orange"]
        color_idxs = np.argmax(self.means @ self.weight_data[k],axis=0) # shapes (4,2) @ (2,4)
        for i in range(4):
            self.weight_arrows.pop(0).remove()  # delete arrow
            self.rejection_arrows.pop(0).remove()
            self.projection_arrows.pop(0).remove()
            warrow = self.ax.arrow(
                0,
                0,
                *self.weight_data[k, :, i],
                length_includes_head=True,
                width=0.01,
                color=colors[color_idxs[i]],#color=(1, 0, 0, 0.5),
            )

            proj, reject = projection_rejection(self.weight_data[k, :, i],self.means[i])
            rej_arrow = self.ax.arrow(
                *proj,
                *reject,
                length_includes_head=True,
                width=0.01,
                color=colors[color_idxs[i]],#rscolor=(1, 0, 0, 0.5),  # semi-transparent green arrow
                alpha=0.5,
            )

            proj_arrow = self.ax.arrow(
                0,
                0,
                *proj,
                length_includes_head=True,
                width=0.01,
                color=colors[color_idxs[i]],#rscolor=(1, 0, 0, 0.5),  # semi-transparent green arrow
                alpha=0.35,
            )
            self.weight_arrows.append(warrow)
            self.rejection_arrows.append(rej_arrow)
            self.projection_arrows.append(proj_arrow)

        self.ax.set_title('Loss={}'.format(self.loss_history[k]))

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.weight_arrows
