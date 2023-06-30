# Copyright (c) 2023, Lukas Behammer
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os

class OrientationException(Exception):
    """
    "No valid orientation found!" Use "coronal", "sagittal" or "transversal".
    """
    pass

def plot_in_orientation(img_data, orientation, slice, cmap="gray", ax=None, **kwargs):
    """
        Plots 3D MRI images in different orientations.

        Examples
        ----------
        >>> from Visualization import plot_in_orientation
        >>> from Import import img_data_loader
        >>> path_T1w = "./Data/S1200_AverageT1w_restore.nii.gz"
        >>> img_T1w, img_data_T1w = img_data_loader(path_T1w)
        >>> plotted_img = plot_in_orientation(img_data=img_data_T1w, orientation="coronal", slice=50,
        >>>                                     title="T1 weighted MRI")

        Parameters
        ----------
        :param img_data: array with image data
        :type img_data: np.ndarray
        :param orientation: orientation to plot, must be "coronal", "sagittal" or "transversal"
        :type orientation: str
        :param slice: image plane orthogonal to orientation to show
        :type slice: int
        :param cmap: matplotlib colormap to use for plotting, defaults to "gray"
        :type cmap: str
        :param ax: Axes element to plot in, optional, defaults to None
        :type ax: matplotlib.axes.Axes or None
        :param kwargs: additional arguments passed to matplotlib imshow
        :return: plotted image
        :rtype: matplotlib.axes.Axes
        :raise: OrientationException when incompatible value for parameter orientation is given
    """
    if orientation == "coronal":
        coronal = np.transpose(img_data, [0, 2, 1])
        coronal = np.rot90(coronal, 1)
        plot = ax.imshow(coronal[:, :, slice], cmap=cmap, **kwargs) if ax else plt.imshow(coronal[:, :, slice],
                                                                                          cmap=cmap)
    elif orientation == "sagittal":
        sagittal = np.transpose(img_data, [1, 2, 0])
        sagittal = np.rot90(sagittal, 1)
        plot = ax.imshow(sagittal[:, :, slice], cmap=cmap, **kwargs) if ax else plt.imshow(sagittal[:, :, slice],
                                                                                           cmap=cmap)
    elif orientation == "transversal":
        transversal = np.transpose(img_data, [0, 1, 2])
        transversal = np.rot90(transversal, 1)
        plot = ax.imshow(transversal[:, :, slice], cmap=cmap, **kwargs) if ax else plt.imshow(transversal[:, :, slice],
                                                                                              cmap=cmap)
    else:
        raise OrientationException
    if not ax:
        plt.show()
    return plot


def create_network_graph_frames(t1w, slices, matrix, all_centroids, regions, region_labels):
    """
        Compute frames for three-dimensional video. Generates images of weighted network graph.

        Examples
        ----------
        >>> from Visualization import create_network_graph_frames
        >>> from Import import get_parcellation_data
        >>> region_maps, region_maps_data, masked_aal, regions, region_labels = get_parcellation_data(fetched=True)
        Atlas has been loaded.
        >>> resampled_img_data_T1w = foo
        >>> correlation_matrices_per_patient = bar
        >>> all_centroids = foobar
        >>> create_network_graph_frames(resampled_img_data_T1w, (50, 45, 45), correlation_matrices_per_patient,
        >>>                                 all_centroids, regions, region_labels)
        All figures saved.

        Parameters
        ----------
        :param t1w: to fMRI data resampled T1 weighted MRI
        :type t1w: np.ndarray
        :param slices: tuple with slice values as int
        :type slices: tuple[int, int, int]
        :param matrix: correlation matrix to be used for edge weight
        :type matrix: np.ndarray
        :param all_centroids: array with coordinates of all centroids
        :type all_centroids: np.ndarray
        :param regions: array with region labels
        :type regions: np.ndarray
        :param region_labels: array with region labels in 3D space
        :type region_labels: np.ndarray
        :return: None
    """
    angle1_part = np.linspace(0, 30, num=int(len(matrix)/4))
    angle1 = np.concatenate((angle1_part, np.flip(angle1_part)[1:], angle1_part[1:], np.flip(angle1_part)[1:],
                             angle1_part[1:]))
    angle2_part = np.linspace(0, 15, num=int(len(matrix) / 4))
    angle2 = np.concatenate((angle2_part, np.flip(angle2_part)[1:], angle2_part[1:], np.flip(angle2_part)[1:],
                             angle2_part[1:]))
    # image transposing
    coronal = np.transpose(t1w, [0, 2, 1])
    coronal = np.rot90(coronal, 3)
    coronal_image = coronal[:, :, slices[0]]
    sagittal = np.transpose(t1w, [1, 2, 0])
    sagittal = np.rot90(sagittal, 0)
    sagittal_image = sagittal[:, :, slices[1]]
    transversal = np.transpose(t1w, [0, 1, 2])
    transversal = np.rot90(transversal, 3)
    transversal_image = transversal[:, :, slices[2]]
    for timestamp in np.arange(len(matrix)):
        graph = nx.from_numpy_array(matrix[timestamp])
        all_weights = []
        for (node1, node2, data) in graph.edges(data=True):
            all_weights.append(matrix[timestamp][node1, node2])
        pos = all_centroids
        node_pos = pos
        edge_pos = np.array([(pos[u], pos[v]) for u, v in graph.edges])
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(projection='3d')
        box = ax.get_position()
        box.x0 = box.x0 - 0.2
        box.x1 = box.x1 - 0.2
        ax.set_position(box)
        x1, y1 = np.meshgrid(np.linspace(0, transversal.shape[1], transversal.shape[1]),
                             np.linspace(0, transversal.shape[0], transversal.shape[0]))
        z1 = np.ones(x1.shape)*-1
        ax.plot_surface(x1, y1, z1, rstride=1, cstride=1,
                        facecolors=plt.cm.gray(transversal_image / transversal_image.max()), shade=False)
        x2, z2 = np.meshgrid(np.linspace(0, coronal.shape[0], coronal.shape[0]),
                             np.linspace(0, coronal.shape[1], coronal.shape[1]))
        y2 = np.ones(x2.shape)*-1
        ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, facecolors=plt.cm.gray(coronal_image / coronal_image.max()),
                        shade=False)
        z3, y3 = np.meshgrid(np.linspace(0, sagittal.shape[1], sagittal.shape[1]),
                             np.linspace(0, sagittal.shape[0], sagittal.shape[0]))
        x3 = np.ones(z3.shape)*-1
        ax.plot_surface(x3, y3, z3, rstride=1, cstride=1, facecolors=plt.cm.gray(sagittal_image / sagittal_image.max()),
                        shade=False)
        scatter = ax.scatter3D(*node_pos.T, c=regions, s=20, cmap="turbo")
        for edge_num in np.arange(edge_pos.shape[0]):
            ax.plot3D(xs=(edge_pos[edge_num, 0, 0], edge_pos[edge_num, 1, 0]),
                      ys=(edge_pos[edge_num, 0, 1], edge_pos[edge_num, 1, 1]),
                      zs=(edge_pos[edge_num, 0, 2], edge_pos[edge_num, 1, 2]), color="red",
                      linewidth=all_weights[edge_num] * 0.5, alpha=0.2)
        ax.view_init(angle2[timestamp], angle1[timestamp])
        legend = fig.legend(handles=scatter.legend_elements(num=len(regions))[0], labels=list(region_labels[:, 0]),
                            bbox_to_anchor=(0.9, 0.5), loc="right", title="regions", ncols=4)
        fig.add_artist(legend)
        ax.set_xlabel("sagittal")
        ax.set_ylabel("transversal")
        ax.set_zlabel("coronal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.savefig(os.path.join("./Data/video/", f"{timestamp}.png"))
        plt.close("all")
    print("All figures saved.")
