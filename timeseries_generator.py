from multiprocessing import Pool
from Main import get_timeseries_per_patient
import itertools


patients = ["100307", "100408", "101107", "101309"]
scan_num = 0
path = "N:/HCP/Unrelated 100/Patients"

if __name__ == '__main__':
    pool = Pool()
    results = pool.starmap(get_timeseries_per_patient, zip(patients, itertools.repeat(scan_num), itertools.repeat(path)))
