# Copyright (c) 2023, Lukas Behammer
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from multiprocessing import Pool
from Main import get_centroids_per_region
from Import import get_parcellation_data
import itertools

region_maps, region_maps_data, masked_aal, regions, region_labels = get_parcellation_data(fetched=True)

regions_from = [0, 29, 58, 87]
regions_to = [29, 58, 87, 116]

if __name__ == '__main__':
    pool = Pool()
    results = pool.starmap(get_centroids_per_region, zip(itertools.repeat(masked_aal), itertools.repeat(regions),
                                                         regions_from, regions_to))
