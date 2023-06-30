# Copyright (c) 2023, Lukas Behammer
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import nibabel as nib
import numpy as np
from nilearn import datasets

def img_data_loader(file):
    """
        Wrapper function to load NIfTI images and extract image data.

        Examples
        ----------
        >>> from Import import img_data_loader
        >>> path = "./Data/S1200_AverageT1w_restore.nii.gz"
        >>> img, img_data = img_data_loader(file=path)

        Parameters
        ----------
        :param file: Path of file to be loaded, defaults to None
        :type file: str or os.PathLike
        :return: img as NIfTI image, img_data as numpy array
    """
    img = nib.load(file)
    # shape = img.header.get_data_shape()
    img_data = img.get_fdata()
    return img, img_data


def get_parcellation_data(parcel_dir='./Data/rois/', fetched=False):
    """
        Wrapper function to fetch or load parcellation atlas (AAL Atlas SPM12) and mask no brain regions.

        Examples
        ----------
        >>> from Import import get_parcellation_data
        >>> get_parcellation_data(fetched=True)
        Atlas has been loaded.

        Parameters
        ----------
        :param parcel_dir: Path to save atlas to, defaults to './Data/rois/'
        :type parcel_dir: str or os.PathLike
        :param fetched: Defines if atlas has already been fetched and should only be loaded from parcel_dir, defaults to
                        False
        :type fetched: boolean
        :return: region_maps as NIfTI image of regions, region_maps_data as numpy array of regions, masked_aal as
            region_maps_data masked to not show "no brain" region, regions as list of all region labels, region_labels
            as region_labels with affiliated region names and short names
    """
    # fetch parcellations
    if fetched is True:
        region_maps = nib.load(os.path.join(parcel_dir, 'aal_SPM12/aal/ROI_MNI_V4.nii'))  # load atlas if already fetched
        print("Atlas has been loaded.")
    else:
        atlas_aal_SPM12 = datasets.fetch_atlas_aal(data_dir=parcel_dir)  # fetch aal atlas
        region_maps = nib.load(atlas_aal_SPM12['maps'])  # get region labels as nifti from atlas
        print("Atlas has been fetched.")
    region_maps_data = region_maps.get_fdata()  # get image data from region labels
    region_labels = np.loadtxt(os.path.join(parcel_dir, 'aal_SPM12/aal/ROI_MNI_V4.txt'), dtype=str)

    # mask regions
    masked_aal = np.ma.masked_where(region_maps_data == 0, region_maps_data)  # mask no brain regions

    # make array in which each region has 0 or 1 for each voxel in 3D space
    regions = np.unique(masked_aal)[0:-1]  # get list of all region labels

    return region_maps, region_maps_data, masked_aal, regions, region_labels
