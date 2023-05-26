from multiprocessing import Pool
from Main import create_network_graph_frames, timeseries_pearson_corr
from Import import img_data_loader, get_parcellation_data
import nilearn.image as nimg
from glob import glob
import numpy as np
import itertools

# setup
patients = ["100307", "100408", "101107", "101309", "101915", "103111", "103414", "103818", "105014", "105115", "106016", "108828", "110411", "111312", "111716", "113619", "113922", "114419", "115320", "116524", "117122", "118528", "118730", "118932", "120111", "122317", "122620", "123117", "123925", "124422", "125525", "126325", "127630", "127933", "128127", "128632", "129028", "130013", "130316", "131217", "131722", "133019", "133928", "135225", "135932", "136833", "138534", "139637", "140925", "144832", "146432", "147737", "148335", "148840", "149337", "149539", "149741", "151223", "151526", "151627", "153025", "154734", "156637", "159340", "160123", "161731", "162733", "163129", "176542", "178950", "188347", "189450", "190031", "192540", "196750", "198451", "199655", "201111", "208226", "211417", "211720", "212318", "214423", "221319", "239944", "245333", "280739", "298051", "366446", "397760", "414229", "499566", "654754", "672756", "751348", "756055", "792564", "856766", "857263", "899885"]
patient_id = "100307"
patient_num = patients.index(patient_id)

sagittal_slice = 70
coronal_slice = 70
transversal_slice = 45

slices = (coronal_slice, sagittal_slice, transversal_slice)

correlation_threshold = 0.75

# import all timeseries
files = glob("D:/HCP/Unrelated 100/Patients/timeseries/*")
all_timeseries = [np.load(file) for file in files]
len(all_timeseries)

# load atlas data
region_maps, region_maps_data, masked_aal, regions, region_labels = get_parcellation_data(fetched=True)

# load Average T1w Image
path_T1w = "./Data/S1200_AverageT1w_restore.nii.gz"
img_T1w, img_data_T1w = img_data_loader(path_T1w)

# import all centroids
files = glob("./Data/centroids/*")
all_centroids = np.concatenate([np.load(file) for file in files], axis=0)

# resample Images to region labels
resampled_img_data_T1w = nimg.resample_to_img(img_T1w, region_maps, interpolation='nearest').get_fdata()

timeseries = all_timeseries[patient_num]
# compute correlation between regions in sliding window
correlation_matrices_per_patient = timeseries_pearson_corr(timeseries, step_width=5, overlap_percentage=0.2)

# set diagonal to 0 (no correlation of region with itself), compute absolute value to have only positive correlations and set all values <= threshold to 0
for k in np.arange(len(correlation_matrices_per_patient)):
    for l in np.arange(len(regions)):
        correlation_matrices_per_patient[k][l,l] = 0
correlation_matrices_per_patient_abs = np.abs(correlation_matrices_per_patient)
correlation_matrices_per_patient_abs_thresh = correlation_matrices_per_patient_abs * (correlation_matrices_per_patient_abs>correlation_threshold)

matrices = [correlation_matrices_per_patient_abs_thresh[0:75,:,:], correlation_matrices_per_patient_abs_thresh[75:150,:,:], correlation_matrices_per_patient_abs_thresh[150:225,:,:], correlation_matrices_per_patient_abs_thresh[225:300,:,:]]

if __name__ == '__main__':
    pool = Pool()
    results = pool.starmap(create_network_graph_frames, zip(itertools.repeat(resampled_img_data_T1w), itertools.repeat(slices), matrices, itertools.repeat(all_centroids)))