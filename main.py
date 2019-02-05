#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os
import sys
import time
import pickle
from pprint import pprint
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from opt import Options
from src.procrustes import get_transformation
import src.data_process as data_process
from src import Bar
import src.utils as utils
import src.misc as misc
import src.log as log

from src.model import LinearModel, weight_init
from src.datasets.human36m import Human36M

import refine_utils as ru


def mpjpe_fun(lhs, rhs):
    """Mean distance of each joints."""
    dist = np.sqrt(np.sum(np.square(lhs - rhs), axis=-1))
    return np.mean(dist)


def pck_fun(lhs, rhs, thresholds):
    """Percentage of correct keypoints."""
    n_joint, _ = lhs.shape
    dist = np.sqrt(np.sum(np.square(lhs - rhs), axis=-1))
    return [
        np.sum(dist <= threshold) / float(n_joint) for threshold in thresholds
    ]


def add_gaussion_noise(arr, mean=0.0, percent=1):
    arr_np = arr.detach().cpu().numpy()
    low = np.percentile(arr_np, 5)
    up = np.percentile(arr_np, 95)
    scale = (up - low) / 2.0 * percent / 100
    return arr + torch.normal(mean, torch.full_like(arr, scale))


def main(opt):
    start_epoch = 0
    err_best = 1000
    glob_step = 0
    lr_now = opt.lr

    manual_seed = 1234
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # save options
    log.save_options(opt, opt.ckpt)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # create model
    print(">>> creating model")
    model = LinearModel()
    model = model.to(device)
    model.apply(weight_init)
    print(">>> total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = nn.MSELoss(size_average=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # load ckpt
    if opt.load:
        print(">>> loading ckpt from '{}'".format(opt.load))
        ckpt = torch.load(opt.load)
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        glob_step = ckpt['step']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt loaded (epoch: {} | err: {})".format(
            start_epoch, err_best))
    if opt.resume:
        logger = log.Logger(os.path.join(opt.ckpt, 'log.txt'), resume=True)
    else:
        logger = log.Logger(os.path.join(opt.ckpt, 'log.txt'))
        logger.set_names(['epoch', 'lr', 'loss_train', 'loss_test', 'err_test'])

    # list of action(s)
    actions = misc.define_actions(opt.action)
    num_actions = len(actions)
    print(">>> actions to use (total: {}):".format(num_actions))
    pprint(actions, indent=4)
    print(">>>")

    # data loading
    print(">>> loading data")
    # load statistics data
    stat_3d = torch.load(os.path.join(opt.data_dir, 'stat_3d.pth.tar'))
    # test
    if opt.test:
        refine_dic, refine_per_action, coeff_funs, refine_extra_kwargs = ru.get_refine_model_config(
            opt)
        pck_thresholds = [50, 100, 150, 200, 250]
        noise_fun = lambda x: add_gaussion_noise(x, percent=opt.noise_level)

        err_set = []
        pck_set = []
        for action in actions:
            print(">>> TEST on _{}_".format(action))
            test_loader = DataLoader(
                dataset=Human36M(
                    actions=action,
                    data_path=opt.data_dir,
                    use_hg=opt.use_hg,
                    is_train=False),
                batch_size=opt.test_batch,
                shuffle=False,
                pin_memory=True)

            refine_idx_action = ru.get_idx_action(action)
            if refine_per_action:
                refine_dic_i = refine_dic[refine_idx_action]
            else:
                refine_dic_i = refine_dic
            coeff_fun_i = coeff_funs[refine_idx_action]
            _, err_test, pck_test = test(
                test_loader,
                model,
                criterion,
                stat_3d,
                device,
                procrustes=opt.procrustes,
                noise_fun=noise_fun,
                pck_thresholds=pck_thresholds,
                refine_dic=refine_dic_i,
                refine_coeff_fun=coeff_fun_i,
                refine_extra_kwargs=refine_extra_kwargs,
                cache_prefix=action if opt.dump_err else None)
            err_set.append(err_test)
            pck_set.append(pck_test)
        print(">>>>>> TEST results:")
        for action in actions:
            print("{}".format(action[:7]), end='\t')
        print("\n")
        for err in err_set:
            print("{:7.1f}".format(err), end='\t')
        print(">>> ERRORS: {}".format(np.array(err_set).mean()))

        for i, thres in enumerate(pck_thresholds):
            for pck in pck_set:
                print("{:7.4f}".format(pck[i]), end='\t')
            print(">>> PCKS {}: {}".format(thres,
                                           np.mean(
                                               [pck[i] for pck in pck_set])))
        sys.exit()

    # load dadasets for training
    test_loader = DataLoader(
        dataset=Human36M(
            actions=actions,
            data_path=opt.data_dir,
            use_hg=opt.use_hg,
            is_train=False),
        batch_size=opt.test_batch,
        shuffle=False,
        num_workers=opt.job,
        pin_memory=True)
    train_loader = DataLoader(
        dataset=Human36M(
            actions=actions, data_path=opt.data_dir, use_hg=opt.use_hg),
        batch_size=opt.train_batch,
        shuffle=True,
        num_workers=opt.job,
        pin_memory=True)
    print(">>> data loaded !")

    cudnn.benchmark = True
    for epoch in range(start_epoch, opt.epochs):
        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))

        # per epoch
        glob_step, lr_now, loss_train = train(
            train_loader,
            model,
            criterion,
            optimizer,
            device,
            lr_init=opt.lr,
            lr_now=lr_now,
            glob_step=glob_step,
            lr_decay=opt.lr_decay,
            gamma=opt.lr_gamma,
            max_norm=opt.max_norm)
        loss_test, err_test, pck_test = test(
            test_loader,
            model,
            criterion,
            stat_3d,
            device,
            procrustes=opt.procrustes)

        # update log file
        logger.append(
            [epoch + 1, lr_now, loss_train, loss_test, err_test, pck_test],
            ['int', 'float', 'float', 'float', 'float', 'float'])

        # save ckpt
        is_best = err_test < err_best
        err_best = min(err_test, err_best)
        if is_best:
            log.save_ckpt({
                'epoch': epoch + 1,
                'lr': lr_now,
                'step': glob_step,
                'err': err_best,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            },
                          ckpt_path=opt.ckpt,
                          is_best=True)
        else:
            log.save_ckpt({
                'epoch': epoch + 1,
                'lr': lr_now,
                'step': glob_step,
                'err': err_best,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            },
                          ckpt_path=opt.ckpt,
                          is_best=False)

    logger.close()


def train(train_loader,
          model,
          criterion,
          optimizer,
          device,
          lr_init=None,
          lr_now=None,
          glob_step=None,
          lr_decay=None,
          gamma=None,
          max_norm=True):
    losses = utils.AverageMeter()

    model.train()

    start = time.time()
    batch_time = 0
    bar = Bar('>>>', fill='>', max=len(train_loader))

    for i, (inps, tars) in enumerate(train_loader):
        glob_step += 1
        if glob_step % lr_decay == 0 or glob_step == 1:
            lr_now = utils.lr_decay(optimizer, glob_step, lr_init, lr_decay,
                                    gamma)
        inputs = Variable(inps.to(device))
        targets = Variable(tars.to(device))

        outputs = model(inputs)

        # calculate loss
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        losses.update(loss.data[0], inputs.size(0))
        loss.backward()
        if max_norm:
            nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()

        # update summary
        if (i + 1) % 100 == 0:
            batch_time = time.time() - start
            start = time.time()

        bar.suffix = '({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.4f}' \
            .format(batch=i + 1,
                    size=len(train_loader),
                    batchtime=batch_time * 10.0,
                    ttl=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg)
        bar.next()

    bar.finish()
    return glob_step, lr_now, losses.avg


def test(test_loader,
         model,
         criterion,
         stat_3d,
         device,
         procrustes=False,
         pck_thresholds=[50, 100, 150, 200, 250],
         noise_fun=lambda x: x,
         refine_dic=None,
         refine_coeff_fun=None,
         refine_extra_kwargs={},
         cache_prefix=None,
         visualize=False):
    model.eval()

    all_outputs = []
    all_targets = []
    losses = utils.AverageMeter()
    for i, (inps, tars) in enumerate(test_loader):
        inps_noise = noise_fun(inps)
        inputs = Variable(inps_noise.to(device))
        targets = Variable(tars.to(device))

        outputs = model(inputs)

        # calculate loss
        outputs_coord = outputs
        loss = criterion(outputs_coord, targets)
        losses.update(loss.item(), inputs.size(0))

        tars = targets

        # calculate erruracy
        targets_unnorm = data_process.unNormalizeData(
            tars.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'],
            stat_3d['dim_use'])
        outputs_unnorm = data_process.unNormalizeData(
            outputs.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'],
            stat_3d['dim_use'])

        # remove dim ignored
        dim_use = np.hstack((np.arange(3), stat_3d['dim_use']))

        outputs_use = outputs_unnorm[:, dim_use]
        targets_use = targets_unnorm[:, dim_use]

        all_outputs.append(outputs_use)
        all_targets.append(targets_use)

    accu_frames = np.cumsum(test_loader.dataset.frames)
    all_outputs = np.split(np.concatenate(all_outputs, axis=0),
                           accu_frames)[:-1]
    all_targets = np.split(np.concatenate(all_targets, axis=0),
                           accu_frames)[:-1]

    start = time.time()
    seq_time = 0
    bar = Bar('>>>', fill='>', max=len(all_outputs))

    all_dist, all_pck = [], []
    for i, (outputs_use, targets_use) in enumerate(
            zip(all_outputs, all_targets)):
        if refine_dic is not None:
            origin = outputs_use
            outputs_use, _ = ru.refine_by_model(outputs_use, refine_dic,
                                                refine_coeff_fun,
                                                **refine_extra_kwargs)

            if visualize:
                visual = [
                    ru.convert_to_pose_16(seq.reshape([-1, 17, 3]))
                    for seq in [outputs_use, origin, targets_use]
                ]
                ru.plot_pose_seq(visual, plot_axis=True, r=1000)

        if procrustes:
            for frame in range(outputs_use.shape[0]):
                gt = targets_use[frame].reshape(-1, 3)
                out = outputs_use[frame].reshape(-1, 3)
                _, Z, T, b, c = get_transformation(
                    gt, out, True, reflection=False)
                out = (b * out.dot(T)) + c
                outputs_use[frame, :] = out.reshape(1, 51)

        for pred, gt in zip(outputs_use, targets_use):
            pred, gt = pred.reshape([-1, 3]), gt.reshape([-1, 3])
            all_dist.append(mpjpe_fun(pred, gt))
            all_pck.append(pck_fun(pred, gt, thresholds=pck_thresholds))

        # update summary
        seq_time = time.time() - start
        start = time.time()

        bar.suffix = '({seq}/{size}) | seq: {seqtime:.4}s | Total: {ttl} | ETA: {eta:} | mpjpe: {loss:.6f}' \
            .format(seq=i + 1,
                    size=len(all_outputs),
                    seqtime=seq_time,
                    ttl=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=np.mean(all_dist))
        bar.next()

    all_dist = np.vstack(all_dist)
    all_pck = np.array(all_pck)
    mpjpe = np.mean(all_dist)
    if cache_prefix:
        with open('cache/{}_.pkl'.format(cache_prefix), 'wb') as f:
            pickle.dump({'mpjpe': all_dist, 'pck': all_pck}, f)
    pck = np.mean(all_pck, axis=0)
    bar.finish()
    print(">>> error: {:4f}, pck: {} <<<".format(
        mpjpe, ' '.join(['{:4f}'.format(val) for val in pck])))
    return losses.avg, mpjpe, pck


if __name__ == "__main__":
    option = Options().parse()
    main(option)
