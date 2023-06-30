# Copyright (c) 2023, Lukas Behammer
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from multiprocessing import Pool
from Main import get_timeseries_per_patient
import itertools


# patients = ["100307", "100408", "101107", "101309", "101915", "103111", "103414", "103818", "105014", "105115", "106016", "108828"]
# patients = ["110411", "111312", "111716", "113619", "113922", "114419", "115320", "116524", "117122", "118528", "118730", "118932"]
# patients = ["120111", "122317", "122620", "123117", "123925", "124422", "125525", "126325", "127630", "127933", "128127", "128632"]
# patients = ["129028", "130013", "130316", "131217", "131722", "133019", "133928", "135225", "135932", "136833", "138534", "139637"]
# patients = ["140925", "144832", "146432", "147737", "148335", "148840", "149337", "149539", "149741", "151223", "151526", "151627"]
# patients = ["153025", "154734", "156637", "159340", "160123", "161731", "162733", "163129", "176542", "178950", "188347", "189450"]
# patients = ["190031", "192540", "196750", "198451", "199655", "201111", "208226", "211417", "211720", "212318", "214423", "221319"]
# patients = ["239944", "245333", "280739", "298051", "366446", "397760", "414229", "499566", "654754", "672756", "751348", "756055"]
patients = ["792564", "856766", "857263", "899885"]
scan_num = 0
path = "N:/HCP/Unrelated 100/Patients"

if __name__ == '__main__':
    pool = Pool()
    results = pool.starmap(get_timeseries_per_patient, zip(patients, itertools.repeat(scan_num), itertools.repeat(path)))
