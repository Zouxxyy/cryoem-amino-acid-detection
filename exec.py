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

"""execution script."""

import argparse
import os
import subprocess
import time

import numpy as np
import torch

import utils.exp_utils as utils
from evaluator import Evaluator
from plotting import plot_batch_prediction

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def train(logger):
    """
    perform the training routine for a given fold. saves plots and selected parameters to the experiment dir
    specified in the configs.
    """
    logger.info('performing training in {}D over fold {} on experiment {} with model {}'.format(
        cf.dim, cf.fold, cf.exp_dir, cf.model))

    net = model.net(cf, logger).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=cf.learning_rate[0], weight_decay=cf.weight_decay)
    model_selector = utils.ModelSelector(cf, logger)
    train_evaluator = Evaluator(cf, logger, mode='train')
    val_evaluator = Evaluator(cf, logger, mode='val_sampling')

    starting_epoch = 1

    # prepare monitoring
    monitor_metrics, TrainingPlot = utils.prepare_monitoring(cf)

    if cf.resume_to_checkpoint:
        starting_epoch, monitor_metrics = utils.load_checkpoint(cf.resume_to_checkpoint, net, optimizer)
        logger.info('resumed to checkpoint {} at epoch {}'.format(cf.resume_to_checkpoint, starting_epoch))

    logger.info('loading dataset and initializing batch generators...')
    batch_gen = data_loader.get_train_generators(cf, logger)

    for epoch in range(starting_epoch, cf.num_epochs + 1):

        logger.info('starting training epoch {}'.format(epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = cf.learning_rate[epoch - 1]

        start_time = time.time()

        net.train()
        train_results_list = []

        for bix in range(cf.num_train_batches):
            batch = next(batch_gen['train'])
            tic_fw = time.time()
            results_dict = net.train_forward(batch)
            tic_bw = time.time()
            optimizer.zero_grad()
            results_dict['torch_loss'].backward()
            optimizer.step()
            logger.info('tr. batch {0}/{1} (ep. {2}) fw {3:.3f}s / bw {4:.3f}s / total {5:.3f}s || '
                        .format(bix + 1, cf.num_train_batches, epoch, tic_bw - tic_fw,
                                time.time() - tic_bw, time.time() - tic_fw) + results_dict['logger_string'])
            train_results_list.append([results_dict['boxes'], batch['pid']])
            monitor_metrics['train']['monitor_values'][epoch].append(results_dict['monitor_values'])

        _, monitor_metrics['train'] = train_evaluator.evaluate_predictions(train_results_list, monitor_metrics['train'])
        train_time = time.time() - start_time

        logger.info('starting validation in mode {}.'.format('val_sampling'))
        with torch.no_grad():
            net.eval()
            if cf.do_validation:
                val_results_list = []
                for _ in range(batch_gen['n_val']):
                    batch = next(batch_gen['val_sampling'])
                    results_dict = net.train_forward(batch, is_validation=True)
                    val_results_list.append([results_dict['boxes'], batch['pid']])
                    monitor_metrics['val']['monitor_values'][epoch].append(results_dict['monitor_values'])

                _, monitor_metrics['val'] = val_evaluator.evaluate_predictions(val_results_list, monitor_metrics['val'])
                model_selector.run_model_selection(net, optimizer, monitor_metrics, epoch)

            # update monitoring and prediction plots
            TrainingPlot.update_and_save(monitor_metrics, epoch)
            epoch_time = time.time() - start_time
            logger.info('trained epoch {}: took {} sec. ({} train / {} val)'.format(
                epoch, epoch_time, train_time, epoch_time - train_time))
            batch = next(batch_gen['val_sampling'])
            results_dict = net.train_forward(batch, is_validation=True)
            logger.info('plotting predictions from validation sampling.')
            plot_batch_prediction(batch, results_dict, cf)


def test(logger):
    batch_gen = data_loader.get_test_generator(cf, logger)
    net = model.net(cf, logger).cuda()
    net.load_state_dict(torch.load(cf.test_weight_path))
    net.eval()
    with torch.no_grad():
        for _ in range(batch_gen['n_test']):
            batch = next(batch_gen['test'])
            results_dict = net.test_forward(batch)
            for index, crop_id in enumerate(batch['pid']):
                logger.info('predict {}, count {} boxes'.format(crop_id, len(results_dict['boxes'][index])))
                with open(os.path.join(cf.pred_dir, crop_id + '_pred.txt'), 'w') as label_file:
                    for dic in results_dict['boxes'][index]:
                        if dic['box_type'] == 'det':
                            x1, y1, x2, y2, z1, z2 = dic['box_coords']
                            label_file.write(str(x1) + ',' + str(x2) + ',' + str(y1) + ',' + str(y2) + ',' +
                                             str(z1) + ',' + str(z2) + ',' + str(dic['box_pred_class_id']) + '\n')
                seg_preds = results_dict['seg_preds'][index][0]
                np.save(os.path.join(cf.pred_dir, crop_id + '_seg_pred'), seg_preds)


if __name__ == '__main__':

    # python -u exec.py --mode train --gpu_id 0 \
    #                     --exp_dir /mnt/data/fzw/amino-acid-detection/exec_dir/${work_dir} \
    #                     --pp_dir /mnt/data/zxy/amino-acid-detection/pp_dir/${dataset}
    # python exec.py --mode train --gpu_id 0 --exp_dir /mnt/data/fzw/amino-acid-detection/exec_dir/0524-0030 --pp_dir /mnt/data/zxy/amino-acid-detection/pp_dir/400_500

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='one out of: train / test')
    parser.add_argument('--pp_dir', type=str, required=True,
                        help='preprocessing dir.')
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='path to experiment dir. will be created if non existent.')
    parser.add_argument('--test_weight_path', type=str,
                        help='weight path for test.')
    parser.add_argument('--test_id_path', type=str,
                        help='id for test.')
    parser.add_argument('--use_stored_settings', default=False, action='store_true',
                        help='load configs from existing exp_dir instead of source dir. always done for testing, '
                             'but can be set to true to do the same for training. useful in job scheduler environment, '
                             'where source code might change before the job actually runs.')
    parser.add_argument('--resume_to_checkpoint', type=str, default=None,
                        help='if resuming to checkpoint, the desired fold still needs to be parsed via --folds.')
    parser.add_argument('--exp_source', type=str, default='experiments/cryoEM_exp',
                        help='specifies, from which source experiment to load configs and data_loader.')
    parser.add_argument('-d', '--dev', default=False, action='store_true', help="development mode: shorten everything")
    parser.add_argument('--gpu_id', default='1', type=str, help="default gpu")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    cf = utils.prep_exp(args)
    model = utils.import_module('model', cf.model_path)
    data_loader = utils.import_module('dl', os.path.join(args.exp_source, 'data_loader.py'))

    if args.mode == 'train':
        for fold in range(cf.n_cv_splits):
            cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(fold))
            cf.fold = fold
            os.makedirs(cf.fold_dir, exist_ok=True)
            logger = utils.get_logger(cf.fold_dir)
            train(logger)
            for hdlr in logger.handlers:
                hdlr.close()
            logger.handlers = []

    elif args.mode == 'test':
        logger = utils.get_logger(cf.exp_dir)
        test(logger)

    else:
        raise RuntimeError('mode specified in args is not implemented...')
