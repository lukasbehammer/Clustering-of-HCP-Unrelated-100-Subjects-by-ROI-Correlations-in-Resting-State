from Import import get_parcellation_data
import nilearn.image as nimg
import nibabel as nib
import numpy as np
import os


def get_timeseries_per_patient(patient_id, scan_num, path="N:/HCP/Unrelated 100/Patients"):
    """
        Calculates timeseries of mean intensity per parcellation region for one patient.

        Examples
        ----------
        >>> from Main import get_timeseries_per_patient
        >>> patient = "100307"
        >>> timeseries_100307 = get_timeseries_per_patient(patient_id=patient, scan_num=3)

        Parameters
        ----------
        :param patient_id: ID of the patient to calculate timeseries of
        :type patient_id: str
        :param scan_num: Number of session, needs to be one of [0, 1, 2, 3]
        :type scan_num: int
        :param path: Location of the patients folders
        :type path: str or os.PathLike
        :return: timeseries with mean intensity per atlas region
    """
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
    """
        Calculates the pearson's r coefficient for a timeseries in a sliding window.

        Examples
        ----------
        >>> from Main import timeseries_pearson_corr
        >>> from Main import get_timeseries_per_patient
        >>> patient = "100307"
        >>> timeseries = np.array(get_timeseries_per_patient(patient_id=patient, scan_num=3))
        >>> timeseries_pearson_corr(timeseries)

        Parameters
        ----------
        :param arr_1: Array with timeseries to calculate correlation of
        :type arr_1: np.ndarray
        :param step_width: Width of sliding window, defaults to 12
        :type step_width: int
        :param overlap_percentage: Percentage of sliding window overlap as decimal
        :type overlap_percentage: float
        :return: correlation matrix with inter-region correlations
    """
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
    """
        Calculates centroid for given list of coordinates.

        Examples
        ----------
        >>> from Main import get_centroid
        >>> coordinates = [[5, 2, 5], [2, 1, 4]]
        >>> centroids = get_centroid(coordinates)
        >>> print(centroids)
        (3.5, 1.5, 4.5)

        Parameters
        ----------
        :param coordinate_list: list of coordinate to compute centroid thereof
        :type coordinate_list: list
        :return: coordinates of centroid
    """
    length = len(coordinate_list)
    sum_x = np.sum([coordinate_list[item][0] for item in np.arange(length)])
    sum_y = np.sum([coordinate_list[item][1] for item in np.arange(length)])
    sum_z = np.sum([coordinate_list[item][2] for item in np.arange(length)])
    return sum_x/length, sum_y/length, sum_z/length


def get_centroids_per_region(region_labels_data, regions, region_from, region_to):
    """
        Calculate centroids for interval of regions.

        Examples
        ----------
        >>> from Main import get_centroids_per_region
        >>> from Import import get_parcellation_data
        >>> region_maps, region_maps_data, masked_aal, regions, region_labels = get_parcellation_data(fetched=True)
        Atlas has been loaded.
        >>> centroids = get_centroids_per_region(region_labels_data=masked_aal, regions=regions, region_from=0,
        >>>                                         region_to=len(regions))

        Parameters
        ----------
        :param region_labels_data: array with region labels in 3D space
        :type region_labels_data: np.ndarray
        :param regions: array with region labels
        :type regions: np.ndarray
        :param region_from: region number to start
        :type region_from: int
        :param region_to: region number to end
        :type region_to: int
        :return: list of regions centroids
    """
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
