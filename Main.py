from Import import get_parcellation_data
import matplotlib.pyplot as plt
import nilearn.image as nimg
import nibabel as nib
import numpy as np
import networkx as nx
import os


def plot_in_orientation(img_data, orientation, slice, cmap="gray", ax=None):
    if orientation == "coronal":
        coronal = np.transpose(img_data, [0, 2, 1])
        coronal = np.rot90(coronal, 1)
        ax.imshow(coronal[:, :, slice], cmap=cmap) if ax else plt.imshow(coronal[:, :, slice], cmap=cmap)
    elif orientation == "sagittal":
        sagittal = np.transpose(img_data, [1, 2, 0])
        sagittal = np.rot90(sagittal, 1)
        ax.imshow(sagittal[:, :, slice], cmap=cmap) if ax else plt.imshow(sagittal[:, :, slice], cmap=cmap)
    elif orientation == "transversal":
        transversal = np.transpose(img_data, [0, 1, 2])
        transversal = np.rot90(transversal, 1)
        ax.imshow(transversal[:, :, slice], cmap=cmap) if ax else plt.imshow(transversal[:, :, slice], cmap=cmap)
    else:
        print("No valid orientation found!")
    if not ax:
        plt.show()


def get_timeseries_per_patient(patient_id, scan_num, path="N:/HCP/Unrelated 100/Patients"):
    # load specific fMRI Image
    scan_names = ["rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL"]
    file_names = ["rfMRI_REST1_LR_hp2000_clean.nii.gz", "rfMRI_REST1_RL_hp2000_clean.nii.gz",
                  "rfMRI_REST2_LR_hp2000_clean.nii.gz", "rfMRI_REST2_RL_hp2000_clean.nii.gz"]
    subpath = f"{patient_id}/MNINonLinear/Results/{scan_names[scan_num]}"
    file_fMRI = os.path.join(path, subpath, file_names[scan_num])
    img_fMRI = nib.load(file_fMRI)

    # load atlas data
    region_maps, region_maps_data, masked_aal, regions, region_labels = get_parcellation_data(fetched=True)

    mean_intensity_per_region_array = []
    shape = img_fMRI.header.get_data_shape()  # get shape for amount of timestamps

    # resample fMRI image to region labels
    resampled_img_fMRI = nimg.resample_to_img(img_fMRI, region_maps, interpolation='nearest') \
        if shape[:3] != region_maps_data.shape else img_fMRI  # HCP fMRI image has already same
    # size as region labels

    # compute activity time series
    for timestamp in np.arange(shape[3]):

        # resample Image
        resampled_img_data_fMRI = resampled_img_fMRI.dataobj[:, :, :, timestamp]  # get data for timestamp

        # calculate mean intensity per region and timestamp
        mean_intensity_per_region_list = []
        for region in regions:
            mean_intensity_per_region_list.append(np.mean(resampled_img_data_fMRI * (region_maps_data == region)))
            # compute mean intensity per region
        mean_intensity_per_region_array.append(mean_intensity_per_region_list)  # add mean intensity per region for this
        # timestamp to time series

    filename = f"patient-{patient_id}_scan-{scan_num}_timeseries.npy"
    file = os.path.join(path, "timeseries", filename)
    np.save(file=file, arr=mean_intensity_per_region_array)  # save time series as numpy object

    return mean_intensity_per_region_array


def timeseries_pearson_corr(arr_1, step_width=12, overlap_percentage=0.2):
    shape = arr_1.shape
    offset = int(overlap_percentage * step_width)
    correlation_matrix = []
    start = 0
    while True:
        stop = start + step_width
        if stop > shape[0]:
            break
        r = np.corrcoef(arr_1[start:stop, :], rowvar=False)
        correlation_matrix.append(r)
        start = stop - offset

    return correlation_matrix


def get_centroid(coordinate_list):
    length = len(coordinate_list)
    sum_x = np.sum([coordinate_list[item][0] for item in np.arange(length)])
    sum_y = np.sum([coordinate_list[item][1] for item in np.arange(length)])
    sum_z = np.sum([coordinate_list[item][2] for item in np.arange(length)])
    return sum_x/length, sum_y/length, sum_z/length


def get_centroids_per_region(region_labels_data, regions, region_from, region_to):
    regions = regions[region_from:region_to]
    centroids = []
    for region in regions:
        coordinate_list = []
        for x in np.arange(region_labels_data.shape[0]):
            for y in np.arange(region_labels_data.shape[1]):
                for z in np.arange(region_labels_data.shape[2]):
                    coordinate_list.append((x, y, z) if region_labels_data[x, y, z] == region else None)
        while None in coordinate_list:
            coordinate_list.remove(None)
        centroids.append(get_centroid(coordinate_list))
    path = "./Data"
    filename = f"centroids-for-regions-{region_from}-{region_to}.npy"
    file = os.path.join(path, filename)
    np.save(file=file, arr=centroids)
    return centroids


def create_network_graph_frames(t1w, slices, matrix, all_centroids):
    angle1_part = np.linspace(0, 30, num=int(len(matrix)/4))
    angle1 = np.concatenate((angle1_part, np.flip(angle1_part)[1:], angle1_part[1:], np.flip(angle1_part)[1:], angle1_part[1:]))
    angle2_part = np.linspace(0, 15, num=int(len(matrix) / 4))
    angle2 = np.concatenate((angle2_part, np.flip(angle2_part)[1:], angle2_part[1:], np.flip(angle2_part)[1:], angle2_part[1:]))
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
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        x1, y1 = np.meshgrid(np.linspace(0, transversal.shape[1], transversal.shape[1]),
                             np.linspace(0, transversal.shape[0], transversal.shape[0]))
        z1 = np.zeros(x1.shape)
        ax.plot_surface(x1, y1, z1, rstride=1, cstride=1,
                        facecolors=plt.cm.gray(transversal_image / transversal_image.max()), shade=False)
        x2, z2 = np.meshgrid(np.linspace(0, coronal.shape[0], coronal.shape[0]),
                             np.linspace(0, coronal.shape[1], coronal.shape[1]))
        y2 = np.zeros(x2.shape)
        ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, facecolors=plt.cm.gray(coronal_image / coronal_image.max()),
                        shade=False)
        z3, y3 = np.meshgrid(np.linspace(0, sagittal.shape[1], sagittal.shape[1]),
                             np.linspace(0, sagittal.shape[0], sagittal.shape[0]))
        x3 = np.zeros(z3.shape)
        ax.plot_surface(x3, y3, z3, rstride=1, cstride=1, facecolors=plt.cm.gray(sagittal_image / sagittal_image.max()),
                        shade=False)
        ax.scatter3D(*node_pos.T, s=20, ec="w")
        for edge_num in np.arange(edge_pos.shape[0]):
            ax.plot3D(xs=(edge_pos[edge_num, 0, 0], edge_pos[edge_num, 1, 0]),
                      ys=(edge_pos[edge_num, 0, 1], edge_pos[edge_num, 1, 1]),
                      zs=(edge_pos[edge_num, 0, 2], edge_pos[edge_num, 1, 2]), color="red",
                      linewidth=all_weights[edge_num] * 0.5, alpha=0.1)
        ax.view_init(angle2[timestamp], angle1[timestamp])
        ax.set_axis_off()
        plt.savefig(os.path.join("./Data/video/", f"{timestamp}.png"))
        plt.close("all")
    print("All figures saved.")
