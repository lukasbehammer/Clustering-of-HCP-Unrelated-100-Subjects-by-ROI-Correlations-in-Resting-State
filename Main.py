from Import import get_parcellation_data
import nilearn.image as nimg
import nibabel as nib
import numpy as np
import os


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
