import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
# import glob

# path = "./Data/*"
# files = glob.glob(path)
# file = files[0]


def img_data_loader(file):
    img = nib.load(file)
    # shape = img.header.get_data_shape()
    img_data = img.get_fdata()
    return img_data


def plot_in_orientation(img_data, orientation, slice):
    img_data = img_data[:, :, :, 0]
    if orientation == "coronal":
        coronal = np.transpose(img_data, [0, 2, 1])
        coronal = np.rot90(coronal, 1)
        plt.imshow(coronal[:, :, slice], cmap="gray")
    elif orientation == "sagittal":
        sagittal = np.transpose(img_data, [1, 2, 0])
        sagittal = np.rot90(sagittal, 1)
        plt.imshow(sagittal[:, :, slice], cmap="gray")
    elif orientation == "transversal":
        transversal = np.transpose(img_data, [0, 1, 2])
        transversal = np.rot90(transversal, 1)
        plt.imshow(transversal[:, :, slice], cmap="gray")
    else:
        print("No valid orientation found!")
    plt.show()
