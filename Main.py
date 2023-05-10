from Import import get_parcellation_data
import nibabel as nib
import numpy as np
import os


def get_timeseries_per_patient(patient, path="./Data/rfMRI_REST1_LR_hp2000_clean.nii.gz"):
    # load specific fMRI Image
    path_fMRI = path  # add os.path.join()
    img_fMRI = nib.load(path_fMRI)

    # load atlas data
    region_labels, masked_aal, regions = get_parcellation_data(fetched=True)

    mean_intensity_per_region_array = []
    shape = img_fMRI.header.get_data_shape()  # get shape for amount of timestamps

    # resample fMRI image to region labels
    # resampled_img_fMRI = nimg.resample_to_img(img_fMRI, region_labels, interpolation='nearest')
    resampled_img_fMRI = img_fMRI  # HCP fMRI image has already same size as region labels

    # compute activity time series
    for timestamp in np.arange(shape[3]):

        # resample Image
        resampled_img_data_fMRI = resampled_img_fMRI.dataobj[:, :, :, timestamp]  # get data for timestamp

        # calculate mean intensity per region and timestamp
        mean_intensity_per_region_list = []
        for region in regions:
            mean_intensity_per_region_list.append(np.mean(resampled_img_data_fMRI[region_labels == region]))
            # compute mean intensity per region
        mean_intensity_per_region_array.append(mean_intensity_per_region_list)  # add mean intensity per region for this
        # timestamp to time series

    filename = f"{patient}_timeseries.npy"
    path = "./Data/timeseries"
    file = os.path.join(path, filename)
    np.save(file, mean_intensity_per_region_array)  # save time series as numpy object

    return mean_intensity_per_region_array


def get_all_timeseries(path):
    # list_of_patients =
    for patient in list_of_patients:
        get_timeseries_per_patient(path, patient)


def timeseries_pearson_corr(arr_1, step_width=12, overlap_percentage=0.2):
    shape = arr_1.shape
    offset = int(overlap_percentage * step_width)
    connectomics = []
    start = 0
    while True:
        stop = start + step_width
        if stop > shape[0]:
            break
        r = np.corrcoef(arr_1[start:stop, :], rowvar=False)
        connectomics.append(r)
        start = stop - offset

    return connectomics
