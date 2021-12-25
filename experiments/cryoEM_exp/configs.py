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

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
from default_configs import DefaultConfigs


class configs(DefaultConfigs):

    def __init__(self):

        self.dim = 3

        # one out of ['retina_net', 'retina_unet', 'ufrcnn', 'detection_unet', 'mrcnn'].
        self.model = 'retina_net'

        DefaultConfigs.__init__(self, self.model, self.dim)

        #########################
        #     preprocessing     #
        #########################

        # path to preprocessed data.
        self.dataset = 'test'
        self.EMdata_dir = '/mnt/data/zxy/amino-acid-detection/EMdata_dir/{}'.format(self.dataset)
        self.pp_dir = '/mnt/data/zxy/amino-acid-detection/pp_dir/{}'.format(self.dataset)
        self.box_size = 64
        self.core_size = 50

        #########################
        #      Data Loader      #
        #########################

        # select modalities from preprocessed data
        self.n_channels = 1

        # patch_size to be used for training
        self.patch_size = [self.box_size, self.box_size, self.box_size]

        #########################
        #      Architecture      #
        #########################

        self.start_filts = 18
        self.end_filts = self.start_filts * 2
        self.res_architecture = 'resnet50'  # 'resnet101' , 'resnet50'
        self.norm = None  # one of None, 'instance_norm', 'batch_norm'
        self.weight_decay = 0

        # one of 'xavier_uniform', 'xavier_normal', or 'kaiming_normal', None (=default = 'kaiming_uniform')
        self.weight_init = None

        #########################
        #  Schedule / Selection #
        #########################

        self.num_epochs = 100
        self.batch_size = 4
        self.num_train_batches = 200
        self.do_validation = True
        self.num_val_batches = 10

        #########################
        #   Testing / Plotting  #
        #########################

        self.test_weight_path = os.path.join('/mnt/data1/zxy/TCIA/exec/exe_dir_1219_0/fold_0/22_best_checkpoint/params.pth')
        self.test_ids = ['3j9d_1_1_1', '3j9d_1_2_2', '3j9d_2_2_2', '3j9d_2_2_1']

        # set the top-n-epochs to be saved for temporal averaging in testing.
        self.save_n_models = 5
        self.test_n_epochs = 5
        # set a minimum epoch number for saving in case of instabilities in the first phase of training.
        self.min_save_thresh = 0

        self.report_score_level = ['patient', 'rois']  # choose list from 'patient', 'rois'
        self.class_dict = {1: 'benign', 2: 'malignant'}  # 0 is background.
        self.patient_class_of_interest = 2  # patient metrics are only plotted for one class.
        self.ap_match_ious = [0.1]  # list of ious to be evaluated for ap-scoring.

        self.model_selection_criteria = ['malignant_ap', 'benign_ap']  # criteria to average over for saving epochs.
        self.min_det_thresh = 0.1  # minimum confidence value to select predictions for evaluation.

        # threshold for clustering predictions together (wcs = weighted cluster scoring).
        # needs to be >= the expected overlap of predictions coming from one model (typically NMS threshold).
        # if too high, preds of the same object are separate clusters.
        self.wcs_iou = 1e-5

        self.plot_prediction_histograms = True
        self.plot_stat_curves = False

        #########################
        #   Add model specifics #
        #########################

        {'detection_unet': self.add_det_unet_configs,
         'mrcnn': self.add_mrcnn_configs,
         'ufrcnn': self.add_mrcnn_configs,
         'retina_net': self.add_mrcnn_configs,
         'retina_unet': self.add_mrcnn_configs,
         }[self.model]()

    def add_det_unet_configs(self):

        self.learning_rate = [1e-4] * self.num_epochs

        # aggregation from pixel perdiction to object scores (connected component). One of ['max', 'median']
        self.aggregation_operation = 'max'

        # max number of roi candidates to identify per batch element and class.
        self.n_roi_candidates = 10 if self.dim == 2 else 30

        # loss mode: either weighted cross entropy ('wce'), batch-wise dice loss ('dice), or the sum of both ('dice_wce')
        self.seg_loss_mode = 'dice_wce'

        # if <1, false positive predictions in foreground are penalized less.
        self.fp_dice_weight = 1

        self.wce_weights = [1, 1, 1]
        self.detection_min_confidence = self.min_det_thresh

        # if 'True', loss distinguishes all classes, else only foreground vs. background (class agnostic).
        self.class_specific_seg_flag = True
        self.num_seg_classes = 21 if self.class_specific_seg_flag else 2
        self.head_classes = self.num_seg_classes

    def add_mrcnn_configs(self):

        # learning rate is a list with one entry per epoch.
        self.learning_rate = [1e-4] * self.num_epochs

        # disable the re-sampling of mask proposals to original size for speed-up.
        # since evaluation is detection-driven (box-matching) and not instance segmentation-driven (iou-matching),
        # mask-outputs are optional.
        self.return_masks_in_val = True
        self.return_masks_in_test = False

        # set number of proposal boxes to plot after each epoch.
        self.n_plot_rpn_props = 5 if self.dim == 2 else 30

        # number of classes for head networks: n_foreground_classes + 1 (background)
        self.head_classes = 21

        # seg_classes hier refers to the first stage classifier (RPN)
        self.num_seg_classes = 2  # foreground vs. background

        # feature map strides per pyramid level are inferred from architecture.
        self.backbone_strides = {'xy': [1, 2, 4, 8], 'z': [1, 2, 4, 8]}

        # anchor scales are chosen according to expected object sizes in data set. Default uses only one anchor scale
        # per pyramid level. (outer list are pyramid levels (corresponding to BACKBONE_STRIDES), inner list are scales per level.)
        self.rpn_anchor_scales = {'xy': [[2], [4], [8], [16]], 'z': [[2], [4], [8], [16]]}

        # choose which pyramid levels to extract features from: P2: 0, P3: 1, P4: 2, P5: 3.
        self.pyramid_levels = [1, 2, 3]

        # number of feature maps in rpn. typically lowered in 3D to save gpu-memory.
        self.n_rpn_features = 512 if self.dim == 2 else 128

        # anchor ratios and strides per position in feature maps.
        self.rpn_anchor_ratios = [0.5, 1, 2]
        self.rpn_anchor_stride = 1

        # Threshold for first stage (RPN) non-maximum suppression (NMS):  LOWER == HARDER SELECTION
        self.rpn_nms_threshold = 0.7 if self.dim == 2 else 0.7

        # loss sampling settings.
        self.rpn_train_anchors_per_image = 6  # per batch element
        self.train_rois_per_image = 6  # per batch element
        self.roi_positive_ratio = 0.5
        self.anchor_matching_iou = 0.7

        # factor of top-k candidates to draw from  per negative sample (stochastic-hard-example-mining).
        # poolsize to draw top-k candidates from will be shem_poolsize * n_negative_samples.
        self.shem_poolsize = 10

        self.pool_size = (7, 7) if self.dim == 2 else (3, 3, 3)
        self.mask_pool_size = (14, 14) if self.dim == 2 else (5, 5, 5)
        self.mask_shape = (28, 28) if self.dim == 2 else (10, 10, 10)

        self.rpn_bbox_std_dev = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
        self.bbox_std_dev = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
        self.window = np.array([0, 0, self.patch_size[0], self.patch_size[1], 0, self.patch_size[2]])
        self.scale = np.array([self.patch_size[0], self.patch_size[1], self.patch_size[0], self.patch_size[1],
                               self.patch_size[2], self.patch_size[2]])

        # pre-selection in proposal-layer (stage 1) for NMS-speedup. applied per batch element.
        self.pre_nms_limit = 3000 if self.dim == 2 else 6000

        # n_proposals to be selected after NMS per batch element. too high numbers blow up memory if "detect_while_training" is True,
        # since proposals of the entire batch are forwarded through second stage in as one "batch".
        self.roi_chunk_size = 2500 if self.dim == 2 else 600
        self.post_nms_rois_training = 500 if self.dim == 2 else 75
        self.post_nms_rois_inference = 500

        # Final selection of detections (refine_detections)
        self.model_max_instances_per_batch_element = 10 if self.dim == 2 else 30  # per batch element and class.
        self.detection_nms_threshold = 1e-5  # needs to be > 0, otherwise all predictions are one cluster.
        self.model_min_confidence = 0.1

        self.backbone_shapes = np.array(
            [[int(np.ceil(self.patch_size[0] / stride)),
              int(np.ceil(self.patch_size[1] / stride)),
              int(np.ceil(self.patch_size[2] / stride_z))]
             for stride, stride_z in zip(self.backbone_strides['xy'], self.backbone_strides['z'])])

        if self.model == 'ufrcnn':
            self.operate_stride1 = True
            self.class_specific_seg_flag = True
            self.num_seg_classes = 3 if self.class_specific_seg_flag else 2
            self.frcnn_mode = True

        if self.model == 'retina_net' or self.model == 'retina_unet' or self.model == 'prob_detector':
            # implement extra anchor-scales according to retina-net publication.
            self.rpn_anchor_scales['xy'] = [[ii[0], ii[0] * (2 ** (1 / 3)), ii[0] * (2 ** (2 / 3))] for ii in
                                            self.rpn_anchor_scales['xy']]
            self.rpn_anchor_scales['z'] = [[ii[0], ii[0] * (2 ** (1 / 3)), ii[0] * (2 ** (2 / 3))] for ii in
                                           self.rpn_anchor_scales['z']]
            self.n_anchors_per_pos = len(self.rpn_anchor_ratios) * 3

            self.n_rpn_features = 256 if self.dim == 2 else 64

            # pre-selection of detections for NMS-speedup. per entire batch.
            self.pre_nms_limit = 10000 if self.dim == 2 else 50000

            # anchor matching iou is lower than in Mask R-CNN according to https://arxiv.org/abs/1708.02002
            self.anchor_matching_iou = 0.5

            # if 'True', seg loss distinguishes all classes, else only foreground vs. background (class agnostic).
            self.num_seg_classes = 21 if self.class_specific_seg_flag else 2

            if self.model == 'retina_unet':
                self.operate_stride1 = True
