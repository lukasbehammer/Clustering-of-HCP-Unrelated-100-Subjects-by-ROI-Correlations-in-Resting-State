# import matplotlib.pyplot as plt

from Import import *
from nilearn import datasets
from nilearn.regions import connected_label_regions
from nilearn import image as nimg
# from nilearn import plotting as nplot
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
resampled_img = nimg.resample_to_img(img, region_labels, interpolation='nearest').get_fdata()

# mask regions
masked_aal = np.ma.masked_where(region_labels_data == 0, region_labels_data)

# plot MRI-image with parcellation
fig, ax = plt.subplots()
plot_in_orientation(resampled_img, 'transversal', 30, ax=ax)
plot_in_orientation(masked_aal, 'transversal', 30, cmap="Paired", ax=ax)
plt.show()

# calculate mean intensity per region
regions = np.unique(masked_aal)[0:-1]
binary_region_list = []
mean_intensity_per_region = []
for region in regions:
    binary_region = masked_aal == region
    binary_region_list.append(binary_region)
    mean_intensity_per_region.append(np.mean(binary_region * resampled_img))

# plot region
which_region = 125
masked_region = np.ma.masked_where((binary_region_list[which_region] * resampled_img) == 0,
                                   (binary_region_list[which_region] * resampled_img))
plot_in_orientation(masked_region, 'transversal', 45)
