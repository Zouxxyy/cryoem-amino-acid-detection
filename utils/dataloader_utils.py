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

import numpy as np


class fold_generator:
    """
    generates splits of indices for a given length of a dataset to perform n-fold cross-validation.
    splits each fold into 3 subsets for training, validation and testing.
    This form of cross validation uses an inner loop test set, which is useful if test scores shall be reported on a
    statistically reliable amount of patients, despite limited size of a dataset.
    If hold out test set is provided and hence no inner loop test set needed, just add test_idxs to the training data in the dataloader.
    This creates straight-forward train-val splits.
    :returns names list: list of len n_splits. each element is a list of len 3 for train_ix, val_ix, test_ix.
    """

    def __init__(self, seed, n_splits, len_data):
        """
        :param seed: Random seed for splits.
        :param n_splits: number of splits, e.g. 5 splits for 5-fold cross-validation
        :param len_data: number of elements in the dataset.
        """
        self.tr_ix = []
        self.val_ix = []
        self.te_ix = []
        self.slicer = None
        self.missing = 0
        self.fold = 0
        self.len_data = len_data
        self.n_splits = n_splits
        self.myseed = seed
        self.boost_val = 0

    def init_indices(self):

        t = list(np.arange(self.l))
        # round up to next splittable data amount.
        split_length = int(np.ceil(len(t) / float(self.n_splits)))
        self.slicer = split_length
        self.mod = len(t) % self.n_splits
        if self.mod > 0:
            # missing is the number of folds, in which the new splits are reduced to account for missing data.
            self.missing = self.n_splits - self.mod

        self.te_ix = t[:self.slicer]
        self.tr_ix = t[self.slicer:]
        self.val_ix = self.tr_ix[:self.slicer]
        self.tr_ix = self.tr_ix[self.slicer:]

    def new_fold(self):

        slicer = self.slicer
        if self.fold < self.missing:
            slicer = self.slicer - 1

        temp = self.te_ix

        # catch exception mod == 1: test set collects 1+ data since walk through both roudned up splits.
        # account for by reducing last fold split by 1.
        if self.fold == self.n_splits - 2 and self.mod == 1:
            temp += self.val_ix[-1:]
            self.val_ix = self.val_ix[:-1]

        self.te_ix = self.val_ix
        self.val_ix = self.tr_ix[:slicer]
        self.tr_ix = self.tr_ix[slicer:] + temp

    def get_fold_names(self):
        names_list = []
        rgen = np.random.RandomState(self.myseed)
        cv_names = np.arange(self.len_data)

        rgen.shuffle(cv_names)
        self.l = len(cv_names)
        self.init_indices()

        for split in range(self.n_splits):
            train_names, val_names, test_names = cv_names[self.tr_ix], cv_names[self.val_ix], cv_names[self.te_ix]
            names_list.append([train_names, val_names, test_names, self.fold])
            self.new_fold()
            self.fold += 1

        return names_list
