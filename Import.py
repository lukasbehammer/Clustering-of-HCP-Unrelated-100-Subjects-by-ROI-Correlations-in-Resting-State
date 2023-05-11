import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.regions import connected_label_regions


def img_data_loader(file):
    img = nib.load(file)
    # shape = img.header.get_data_shape()
    img_data = img.get_fdata()
    return img, img_data


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


def get_parcellation_data(parcel_dir='./resources/rois/', fetched=False):
    # fetch parcellations
    if fetched is True:
        region_labels = nib.load('./resources/rois/aal_SPM12/aal/ROI_MNI_V4.nii')  # load atlas if already fetched
    else:
        atlas_aal_SPM12 = datasets.fetch_atlas_aal(data_dir=parcel_dir)  # fetch aal atlas
        region_labels = connected_label_regions(atlas_aal_SPM12['maps'])  # get region labels as nifti from atlas
    region_labels_data = region_labels.get_fdata()  # get image data from region labels

    # mask regions
    masked_aal = np.ma.masked_where(region_labels_data == 0, region_labels_data)  # mask no brain regions

    # make array in which each region has 0 or 1 for each voxel in 3D space
    regions = np.unique(masked_aal)[0:-1]  # get list of all region labels

    return region_labels, region_labels_data, masked_aal, regions
