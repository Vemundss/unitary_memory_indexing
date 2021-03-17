"""
Code initially taken from (17/03-2021):
https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""

    def __init__(self, cluster_data, weight_data):
        """
        Args:
            cluster_data: (N,2), where N=4*n -> four clusters
            weight_data: (#epochs,2,4)
        """
        self.cluster_data = cluster_data
        self.weight_data = weight_data
        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.animation = animation.FuncAnimation(
            self.fig,
            self.update,
            init_func=self.setup_plot,
            interval=25, # time in ms between frames
            repeat_delay=1000, # delay before loop
            blit=False, # for OSX
            save_count=weight_data.shape[0], # #frames
        )

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        init_weights = self.weight_data
        data = self.cluster_data
        colors = ["red", "green", "blue", "orange"]
        means = np.mean(
            np.reshape(data, (4, int(data.shape[0] / 4), data.shape[-1])), axis=1
        )
        cluster_N = int(data.shape[0] / 4)
        self.weight_arrows = []
        for color, mean, i in zip(colors, means, range(len(colors))):
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
                *init_weights[0, :, i],
                length_includes_head=True,
                width=0.01,
                color=(1, 0, 0, 0.5),  # semi-transparent green arrow
            )
            self.weight_arrows.append(warrow)
        self.ax.grid("on")

    def update(self, k):
        """Update the scatter plot."""
        for i in range(4):
            self.weight_arrows.pop(0).remove()  # delete arrow
            warrow = self.ax.arrow(
                0,
                0,
                *self.weight_data[k, :, i],
                length_includes_head=True,
                width=0.01,
                color=(1, 0, 0, 0.5),
            )
            self.weight_arrows.append(warrow)

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.weight_arrows
