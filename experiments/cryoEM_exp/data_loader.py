#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

'''
Example Data Loader for the LIDC data set. This dataloader expects preprocessed data in .npy or .npz files per patient and
a pandas dataframe in the same directory containing the meta-info e.g. file paths, labels, foregound slice-ids.
'''
import os
import pickle

import numpy as np
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase

import utils.dataloader_utils as dutils


def get_train_generators(cf, logger):
    all_pids_list = []
    for file in os.listdir(cf.pp_dir):
        if file.split('.')[1] == 'npy':
            all_pids_list.append(file.split('.')[0])

    if not cf.created_fold_id_pickle:
        fg = dutils.fold_generator(seed=cf.seed, n_splits=cf.n_cv_splits, len_data=len(all_pids_list)).get_fold_names()
        with open(os.path.join(cf.exp_dir, 'fold_ids.pickle'), 'wb') as handle:
            pickle.dump(fg, handle)
        cf.created_fold_id_pickle = True
    else:
        with open(os.path.join(cf.exp_dir, 'fold_ids.pickle'), 'rb') as handle:
            fg = pickle.load(handle)

    train_ix, val_ix, test_ix, _ = fg[cf.fold]

    train_pids = [all_pids_list[ix] for ix in train_ix]
    train_pids += [all_pids_list[ix] for ix in test_ix]
    val_pids = [all_pids_list[ix] for ix in val_ix]

    logger.info("data set loaded with: {} train / {} val".format(len(train_ix) + len(test_ix), len(val_ix)))

    batch_gen = {'train': BatchGenerator(train_pids, batch_size=cf.batch_size, cf=cf, is_train=True),
                 'val_sampling': BatchGenerator(val_pids, batch_size=cf.batch_size, cf=cf, is_train=True),
                 'n_val': cf.num_val_batches}
    return batch_gen


def get_test_generator(cf, logger):
    test_ids = cf.test_ids
    logger.info("data set loaded with: {} test patients".format(len(test_ids)))
    batch_gen = {'test': BatchGenerator(test_ids, batch_size=1, cf=cf, is_train=False),
                 'n_test': len(test_ids)}
    return batch_gen


class BatchGenerator(SlimDataLoaderBase):

    def __init__(self, data, batch_size, cf, is_train):
        super(BatchGenerator, self).__init__(data, batch_size)
        self.cf = cf
        self.is_train = is_train
        self.cur_ix = -1

    def generate_train_batch(self):
        batch_data, batch_pids, batch_roi_labels, batch_bb_target = [], [], [], []

        train_pids = self._data
        if self.is_train:
            batch_ixs = np.random.choice(len(train_pids), self.batch_size)
        else:
            self.cur_ix += 1
            batch_ixs = [self.cur_ix]

        for b in batch_ixs:
            if b >= len(train_pids):
                continue
            train_pid = train_pids[b]
            data = np.load(os.path.join(self.cf.pp_dir, train_pid + '.npy'), mmap_mode='r')[np.newaxis]  # (c, x, y, z)
            batch_pids.append(train_pid)
            batch_data.append(data)
            rois = []
            labels = []
            with open(os.path.join(self.cf.pp_dir, train_pid + '.txt'), 'r') as label_file:
                for line in label_file:
                    [x1, x2, y1, y2, z1, z2, amino_acid_id] = line.strip('\n').split(',')
                    rois.append(np.array([int(x1), int(y1), int(x2), int(y2), int(z1), int(z2)]))
                    labels.append(int(amino_acid_id))
            batch_roi_labels.append(np.array(labels))
            batch_bb_target.append(np.array(rois))

        data = np.array(batch_data)
        roi_labels = np.array(batch_roi_labels)
        bb_target = np.array(batch_bb_target)
        return {'data': data, 'pid': batch_pids, 'roi_labels': roi_labels, 'bb_target': bb_target}
