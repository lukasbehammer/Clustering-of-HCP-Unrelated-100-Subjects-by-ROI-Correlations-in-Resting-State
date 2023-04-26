import matplotlib.pyplot as plt

from Import import *
from nilearn import datasets
from nilearn.regions import connected_label_regions
from nilearn import image as nimg
from nilearn import plotting as nplot
from glob import glob

# fetch parcellations
parcel_dir = './resources/rois/'
# atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011(data_dir=parcel_dir)
atlas_aal_SPM12 = datasets.fetch_atlas_aal(data_dir=parcel_dir)
# region_labels = connected_label_regions(atlas_yeo_2011['thin_7'])
region_labels = connected_label_regions(atlas_aal_SPM12['maps'])
region_labels_data = region_labels.get_fdata()

# load Average T1w Image
path = "./Data/*"
files = glob(path)
file = files[0]
img, img_data = img_data_loader(file)

# resample image
resampled_aal = nimg.resample_to_img(region_labels, img, interpolation='nearest').get_fdata()

# mask regions
masked_resampled_aal = np.ma.masked_where(resampled_aal==0, resampled_aal)

# plot MRI-image with parcellation
fig, ax = plt.subplots()
plot_in_orientation(img_data, 'transversal', 150, ax=ax)
plot_in_orientation(masked_resampled_aal, 'transversal', int(45*(260/91)), cmap="Paired", ax=ax)
plt.show()

# plot_in_orientation(region_labels_data, 'transversal', 45, cmap="Paired")
