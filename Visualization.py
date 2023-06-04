import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os


def plot_in_orientation(img_data, orientation, slice, cmap="gray", ax=None, **kwargs):
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
        print("No valid orientation found!")
    if not ax:
        plt.show()
    return plot


def create_network_graph_frames(t1w, slices, matrix, all_centroids, regions, region_labels):
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
