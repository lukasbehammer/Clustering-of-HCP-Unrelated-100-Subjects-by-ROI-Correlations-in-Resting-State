from Import import *
from nilearn import datasets
from nilearn.regions import connected_label_regions
from nilearn import image as nimg
from nilearn import plotting as nplot

parcel_dir = './resources/rois/'
atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011(parcel_dir)
region_labels = connected_label_regions(atlas_yeo_2011['thin_7'])
region_labels_data = region_labels.get_fdata()
img_data = img_data_loader(atlas_yeo_2011['thin_7'])
plot_in_orientation(img_data, 'transversal', 150)
plot_in_orientation(region_labels_data, 'transversal', 150)
