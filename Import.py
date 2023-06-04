import nibabel as nib
import numpy as np
from nilearn import datasets


def img_data_loader(file):
    img = nib.load(file)
    # shape = img.header.get_data_shape()
    img_data = img.get_fdata()
    return img, img_data


def get_parcellation_data(parcel_dir='./Data/rois/', fetched=False):
    # fetch parcellations
    if fetched is True:
        region_maps = nib.load('./Data/rois/aal_SPM12/aal/ROI_MNI_V4.nii')  # load atlas if already fetched
    else:
        atlas_aal_SPM12 = datasets.fetch_atlas_aal(data_dir=parcel_dir)  # fetch aal atlas
        region_maps = nib.load(atlas_aal_SPM12['maps'])  # get region labels as nifti from atlas
    region_maps_data = region_maps.get_fdata()  # get image data from region labels
    region_labels = np.loadtxt('./Data/rois/aal_SPM12/aal/ROI_MNI_V4.txt', dtype=str)

    # mask regions
    masked_aal = np.ma.masked_where(region_maps_data == 0, region_maps_data)  # mask no brain regions

    # make array in which each region has 0 or 1 for each voxel in 3D space
    regions = np.unique(masked_aal)[0:-1]  # get list of all region labels

    return region_maps, region_maps_data, masked_aal, regions, region_labels
