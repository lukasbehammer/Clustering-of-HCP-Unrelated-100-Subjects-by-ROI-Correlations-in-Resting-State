from Import import *
from nilearn import datasets
from nilearn.regions import connected_label_regions
from nilearn import image as nimg
import numpy as np

# fetch parcellations
parcel_dir = './resources/rois/'
atlas_aal_SPM12 = datasets.fetch_atlas_aal(data_dir=parcel_dir)
region_labels = connected_label_regions(atlas_aal_SPM12['maps'])
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
for timestamp in np.arange(shape[3]):

    # resample Image
    resampled_img_data_fMRI = nimg.resample_to_img(img_fMRI, region_labels, interpolation='nearest').dataobj[:, :, :, timestamp]

    # calculate mean intensity per region
    mean_intensity_per_region_list = []
    for region in regions:
        mean_intensity_per_region_list.append(np.mean(resampled_img_data_fMRI[binary_region_list[int(region)-1]]))
    mean_intensity_per_region_array.append(mean_intensity_per_region_list)


# # load Average T1w Image
# path_T1w = "./Data/S1200_AverageT1w_restore.nii.gz"
# img_T1w, img_data_T1w = img_data_loader(path_T1w)
#
# # resample T1w Image
# resampled_img_data_T1w = nimg.resample_to_img(img_T1w, region_labels, interpolation='nearest').get_fdata()
#
# # plot fMRI-image with parcellation
# fig, ax = plt.subplots()
# plot_in_orientation(resampled_img_fMRI, 'transversal', 30, ax=ax, cmap="coolwarm")
# plot_in_orientation(masked_aal, 'transversal', 30, cmap="Paired", ax=ax)
# plt.show()
#
# # plot region
# which_region = 125
# masked_region = np.ma.masked_where((binary_region_list[which_region] * resampled_img_data_T1w) == 0,
#                                    (binary_region_list[which_region] * resampled_img_data_T1w))
# plot_in_orientation(masked_region, 'transversal', 45)
