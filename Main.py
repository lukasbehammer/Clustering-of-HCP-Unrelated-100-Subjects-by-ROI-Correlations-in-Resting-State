from Import import *
from nilearn import datasets
from nilearn.regions import connected_label_regions
from nilearn import image as nimg
import nibabel as nib
import numpy as np

# fetch parcellations
parcel_dir = './resources/rois/'
atlas_aal_SPM12 = datasets.fetch_atlas_aal(data_dir=parcel_dir)
region_labels = connected_label_regions(atlas_aal_SPM12['maps'])
# region_labels = nib.load('./resources/rois/aal_SPM12/aal/ROI_MNI_V4.nii')
region_labels_data = region_labels.get_fdata()

# mask regions
masked_aal = np.ma.masked_where(region_labels_data == 0, region_labels_data)

# load specific fMRI Image
path_fMRI = "./Data/rfMRI_REST1_LR_hp2000_clean.nii.gz"
img_fMRI = nib.load(path_fMRI)

# make array in which each region has 0 or 1 for each voxel in 3D space
regions = np.unique(masked_aal)[0:-1]
binary_region_list = []
for region in regions:
    binary_region = masked_aal == region
    binary_region_list.append(binary_region)

mean_intensity_per_region_array = []
shape = img_fMRI.header.get_data_shape()
# resampled_img_fMRI = nimg.resample_to_img(img_fMRI, region_labels, interpolation='nearest')
resampled_img_fMRI = img_fMRI
for timestamp in np.arange(shape[3]):

    # resample Image
    resampled_img_data_fMRI = resampled_img_fMRI.dataobj[:, :, :, timestamp]

    # calculate mean intensity per region
    mean_intensity_per_region_list = []
    for region in regions:
        mean_intensity_per_region_list.append(np.mean(resampled_img_data_fMRI[binary_region_list[int(region)-1]]))
    mean_intensity_per_region_array.append(mean_intensity_per_region_list)

np.save('./Data/mean_intensity_per_region.npy', mean_intensity_per_region_array)
