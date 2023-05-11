from Import import get_parcellation_data
import nilearn.image as nimg
import nibabel as nib
import numpy as np
import os


def get_timeseries_per_patient(patient_id, scan_num, path="./Data/rfMRI_REST1_LR_hp2000_clean.nii.gz"):
    # load specific fMRI Image
    scan_names = ["rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL"]
    file_names = ["rfMRI_REST1_LR_hp2000_clean.nii.gz", "rfMRI_REST1_RL_hp2000_clean.nii.gz",
                  "rfMRI_REST2_LR_hp2000_clean.nii.gz", "rfMRI_REST2_RL_hp2000_clean.nii.gz"]
    subpath = f"{patient_id}/MINNonLinear/Results/{scan_names[scan_num]}"
    file_fMRI = os.path.join(path, subpath, file_names[scan_num])
    img_fMRI = nib.load(file_fMRI)

    # load atlas data
    region_labels, region_labels_data, masked_aal, regions = get_parcellation_data(fetched=True)

    mean_intensity_per_region_array = []
    shape = img_fMRI.header.get_data_shape()  # get shape for amount of timestamps

    # resample fMRI image to region labels
    resampled_img_fMRI = nimg.resample_to_img(img_fMRI, region_labels, interpolation='nearest') \
        if shape[:3] != region_labels_data.shape else img_fMRI  # HCP fMRI image has already same
    # size as region labels

    # compute activity time series
    for timestamp in np.arange(shape[3]):

        # resample Image
        resampled_img_data_fMRI = resampled_img_fMRI.dataobj[:, :, :, timestamp]  # get data for timestamp

        # calculate mean intensity per region and timestamp
        mean_intensity_per_region_list = []
        for region in regions:
            mean_intensity_per_region_list.append(np.mean(resampled_img_data_fMRI * (region_labels_data == region)))
            # compute mean intensity per region
        mean_intensity_per_region_array.append(mean_intensity_per_region_list)  # add mean intensity per region for this
        # timestamp to time series

    filename = f"patient-{patient_id}_scan-{scan_num}_timeseries.npy"
    path = "./Data/timeseries"
    file = os.path.join(path, filename)
    np.save(file, mean_intensity_per_region_array)  # save time series as numpy object

    return mean_intensity_per_region_array


def get_all_timeseries(scans, patients, path="D:/HCP/Unrelated 100/Patients"):
    """
    Gets all timeseries of chosen scans for chosen patients.

    Examples
    --------
    >>> from Main import get_all_timeseries
    >>> scans = [0, 1]
    >>> patients = ["100307", "100408"]
    >>> get_all_timeseries(scans=scans, patients=patients)
    4 timeseries saved as .npy


    Parameters
    ----------
    :param scans: list, default=None
                which scans should be used, four possible scans are '0', '1', '2' and '3', has to be a list like [0, 1]
    :param patients: list, default=None
                which patients to get the scans from, list with HCP patient IDs as strings
    :param path: str or path-like object, default="D:/HCP/Unrelated 100/Patients"
                path where patients folders are located
    :return: None, saves timeseries to numpy file
    """
    num_of_series = 0
    for scan in scans:
        for patient in patients:
            get_timeseries_per_patient(patient_id=patient, scan_num=scan, path=path)
            num_of_series += 1
    print(f"{int(num_of_series)} timeseries saved as .npy")

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
