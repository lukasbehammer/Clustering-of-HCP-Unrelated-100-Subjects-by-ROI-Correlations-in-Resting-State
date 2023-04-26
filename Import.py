import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
# import glob




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
