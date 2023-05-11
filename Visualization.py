from Import import *
from nilearn import image as nimg
import numpy as np

# load atlas data
region_labels, region_labels_data, masked_aal, regions = get_parcellation_data(fetched=True)

# load specific fMRI Image
path_fMRI = "./Data/rfMRI_REST1_LR_hp2000_clean.nii.gz"
img_fMRI = nib.load(path_fMRI)

# load Average T1w Image
path_T1w = "./Data/S1200_AverageT1w_restore.nii.gz"
img_T1w, img_data_T1w = img_data_loader(path_T1w)

# resample Images to region labels
resampled_img_data_T1w = nimg.resample_to_img(img_T1w, region_labels, interpolation='nearest').get_fdata()
# resampled_img_fMRI = nimg.resample_to_img(img_fMRI, region_labels, interpolation='nearest')
resampled_img_fMRI = img_fMRI  # HCP fMRI image has already same size as region labels
timestamp = 100
resampled_img_data_fMRI = resampled_img_fMRI.dataobj[:, :, :, timestamp]

which_region = 12
masked_aal = np.ma.masked_where(masked_aal != regions[which_region], masked_aal)

# plot fMRI-image with parcellation
orientation = 'transversal'
slice = 45
fig, ax = plt.subplots()
plot_in_orientation(resampled_img_data_fMRI, orientation, slice, ax=ax, cmap="coolwarm")
plot_in_orientation(masked_aal, orientation, slice, cmap="Paired", ax=ax)
plt.show()

# plot region
resampled_img_data_fMRI_oneRegionOnly = resampled_img_data_fMRI * (region_labels_data == regions[which_region])
masked_region = np.ma.masked_where(resampled_img_data_fMRI_oneRegionOnly == 0,
                                   resampled_img_data_fMRI_oneRegionOnly)
plot_in_orientation(masked_region, orientation, slice)

mean = np.mean(resampled_img_data_fMRI_oneRegionOnly)
# test = region_labels_data == regions[0]
