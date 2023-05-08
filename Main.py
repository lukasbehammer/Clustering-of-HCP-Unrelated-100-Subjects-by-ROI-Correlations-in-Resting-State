from nilearn import datasets
from nilearn.regions import connected_label_regions
import nibabel as nib
import numpy as np

# fetch parcellations
parcel_dir = './resources/rois/'    # set directory for atlas
atlas_aal_SPM12 = datasets.fetch_atlas_aal(data_dir=parcel_dir) # fetch aal atlas
region_labels = connected_label_regions(atlas_aal_SPM12['maps'])    # get region labels as nifti from atlas
# region_labels = nib.load('./resources/rois/aal_SPM12/aal/ROI_MNI_V4.nii') # load atlas if already fetched
region_labels_data = region_labels.get_fdata()  # get image data from region labels

# mask regions
masked_aal = np.ma.masked_where(region_labels_data == 0, region_labels_data)    # mask no brain regions

# load specific fMRI Image
path_fMRI = "./Data/rfMRI_REST1_LR_hp2000_clean.nii.gz"
img_fMRI = nib.load(path_fMRI)

# make array in which each region has 0 or 1 for each voxel in 3D space
regions = np.unique(masked_aal)[0:-1]   # get list of all region labels
binary_region_list = []
for region in regions:
    binary_region = masked_aal == region    # compute boolean value for each voxel per region
    binary_region_list.append(binary_region)

mean_intensity_per_region_array = []
shape = img_fMRI.header.get_data_shape()    # get shape for amount of timestamps

# resample fMRI image to region labels
# resampled_img_fMRI = nimg.resample_to_img(img_fMRI, region_labels, interpolation='nearest')
resampled_img_fMRI = img_fMRI   # HCP fMRI image has already same size as region labels

# compute activity time series
for timestamp in np.arange(shape[3]):

    # resample Image
    resampled_img_data_fMRI = resampled_img_fMRI.dataobj[:, :, :, timestamp]    # get data for timestamp

    # calculate mean intensity per region and timestamp
    mean_intensity_per_region_list = []
    for region in regions:
        mean_intensity_per_region_list.append(np.mean(resampled_img_data_fMRI[binary_region_list[int(region)-1]]))  #
        # compute mean intensity per region
    mean_intensity_per_region_array.append(mean_intensity_per_region_list)  # add mean intensity per region for this
    # timestamp to time series

np.save('./Data/mean_intensity_per_region.npy', mean_intensity_per_region_array)    # save time series as numpy object
