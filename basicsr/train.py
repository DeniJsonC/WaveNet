import argparse
import datetime
import logging
import math
import random
import time
import torch
from torchvision import transforms
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse

import numpy as np
########data_aug#############
def random_data_aug(lp,gt):
    aug = random.randint(0, 7)
    if aug==0:
        return lp,gt
    elif aug == 1:
        lp = lp.flip(2)
        gt = gt.flip(2)
    elif aug == 2:
        lp = lp.flip(3)
        gt = gt.flip(3)
    elif aug == 3:
        lp = torch.rot90(lp, dims=(2, 3))
        gt = torch.rot90(gt, dims=(2, 3))
    elif aug == 4:
        lp = torch.rot90(lp, dims=(2, 3), k=2)
        gt = torch.rot90(gt, dims=(2, 3), k=2)
    elif aug == 5:
        lp = torch.rot90(lp, dims=(2, 3), k=3)
        gt = torch.rot90(gt, dims=(2, 3), k=3)
    elif aug == 6:
        lp = torch.rot90(lp.flip(2), dims=(2, 3))
        gt = torch.rot90(gt.flip(2), dims=(2, 3))
    elif aug == 7:
        lp = torch.rot90(lp.flip(3), dims=(2, 3))
        gt = torch.rot90(gt.flip(3), dims=(2, 3))
    else:
        raise Exception('Invalid choice of image transformation')
    return lp,gt

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (opt['logger'].get('wandb')
            is not None) and (opt['logger']['wandb'].get('project')
                              is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_epochs=opt['train']['total_epochs']
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}')

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, 


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # automatic resume ..
    state_folder_path = 'experiments/{}/training_states/'.format(opt['name'])
    import os
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []

    resume_state = None
    if len(states) > 0:
        max_state_file = '{}.state'.format(max([int(x[0:-6]) for x in states]))
        # max_state_file='BEST_PSNR_23.15_SSIM_0.72.state'
        resume_state = os.path.join(state_folder_path, max_state_file)
        opt['path']['resume_state'] = resume_state

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs,  = result

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()
    ##################
    mini_gt_sizes=[]
    scale_t=opt['datasets']['train']
    if scale_t['sort']==0:
        for i in range(scale_t['min_patch_size'],scale_t['min_patch_size']+scale_t['stride']*scale_t['step']+1,scale_t['stride']):
            mini_gt_sizes.append(i)
        scale_t['batch_sizes'].sort(reverse=True)
        mini_batch_sizes=scale_t['batch_sizes']
    else:
        for i in range(scale_t['min_patch_size']+scale_t['stride']*scale_t['step'],scale_t['min_patch_size']-1,-scale_t['stride']):
            mini_gt_sizes.append(i)
        mini_batch_sizes=scale_t['batch_sizes']

    data_aug=opt['datasets']['train'].get('geometric_augs')
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')

    scale = opt['scale']
    best_score=0
    patient_epochs=0
    epoch = start_epoch
    #####visualize##########
    # if opt.get('val') is not None and epoch>0:
    #     rgb2bgr = opt['val'].get('rgb2bgr', True)
    #     # wheather use uint8 image to compute metrics
    #     use_image = opt['val'].get('use_image', True)
    #     metric_results=model.validation(val_loader, current_iter, tb_logger,
    #                     opt['val']['save_img'], rgb2bgr, use_image )
    # return 0
    #####################
    while epoch < total_epochs:
        for bs_j in range (len(mini_batch_sizes)):
            for m in range(2):
                scale_t['patience']+=1
                if m%2==0:
                    state_flag=True
                else:
                    state_flag=False
                # state_flag=True
                # scale_t['patience']+=1
                logger.info(f'''==> Training details:
                    ------------------------------------------------------------------
                        Restoration mode:   {opt['name']}
                        Train patches size: {str(mini_gt_sizes[bs_j]) + 'x' + str(mini_gt_sizes[bs_j])}
                        Val patches size:   {str(mini_gt_sizes[bs_j]) + 'x' + str(mini_gt_sizes[bs_j])}
                        Start/End epochs:   {str(epoch) + '~' + str(total_epochs)}
                        Batch sizes:        {mini_batch_sizes[bs_j]}
                        Crop states:        {state_flag}
                        Learning rate:      {model.get_current_learning_rate()}
                        current patience:          {scale_t['patience']}''')
                logger.info('------------------------------------------------------------------')
                for e in range(epoch, total_epochs + 1):
                    train_sampler.set_epoch(epoch)
                    prefetcher.reset()
                    train_data = prefetcher.next()
                    while train_data is not None:
                        data_time = time.time() - data_time
                        current_iter += 1

                        ### ------Multi-scale Training Strategy ---------------------
                        mini_gt_size = mini_gt_sizes[bs_j]
                        mini_batch_size = mini_batch_sizes[bs_j]   
                        lq = train_data['lq']
                        gt = train_data['gt']
                        # gt=gt.repeat(1,2,1,1)
                        if mini_batch_size < batch_size:
                            indices = random.sample(range(0, batch_size), k=mini_batch_size)
                            lq = lq[indices]
                            gt = gt[indices]
                        # crop form original images
                        if state_flag:
                            x0 = int((lq.shape[2] - mini_gt_size) * random.random())
                            y0 = int((lq.shape[3] - mini_gt_size) * random.random())
                            x1 = x0 + mini_gt_size
                            y1 = y0 + mini_gt_size
                            lq = lq[:,:,x0:x1,y0:y1]
                            gt = gt[:,:,x0*scale:x1*scale,y0*scale:y1*scale]    
                        # resize form original images
                        else:
                            lq=transforms.Resize((mini_gt_size,mini_gt_size))(lq)
                            gt=transforms.Resize((mini_gt_size,mini_gt_size))(gt)
                        ###data_aug-------------------------------------------
                        if data_aug:
                            lq,gt=random_data_aug(lq,gt)

                        model.feed_train_data({'lq': lq, 'gt':gt})
                        model.optimize_parameters(current_iter)
                        iter_time = time.time() - iter_time
                        # log
                        if current_iter % opt['logger']['print_freq'] == 0:
                            log_vars = {'epoch': epoch, 'iter': current_iter}
                            log_vars.update({'lrs': model.get_current_learning_rate()})
                            log_vars.update({'time': iter_time, 'data_time': data_time})
                            log_vars.update(model.get_current_log())
                            msg_logger(log_vars)
                        # save models and training states
                        if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                            logger.info('Saving models and training states.')
                            model.save(epoch, current_iter)
                        # validation
                        if opt.get('val') is not None and (current_iter %
                                                        opt['val']['val_freq'] == 0):
                            rgb2bgr = opt['val'].get('rgb2bgr', True)
                            # wheather use uint8 image to compute metrics
                            use_image = opt['val'].get('use_image', True)
                            metric_results=model.validation(val_loader, current_iter, tb_logger,
                                            opt['val']['save_img'], rgb2bgr, use_image )
                            psnr_val_rgb=metric_results[0]
                            ssim_val_rgb=metric_results[1]
                            if ssim_val_rgb*10+psnr_val_rgb>best_score:
                                best_score=ssim_val_rgb*10+psnr_val_rgb
                                patient_epochs=0
                                best_name='BEST_PSNR_'+str(round(psnr_val_rgb,2))+'_SSIM_'+str(round(ssim_val_rgb,2))
                                model.save(epoch, best_name)
                                logger.info('current best psnr: %.4f and ssim : %.4f'%(psnr_val_rgb,ssim_val_rgb))
                                logger.info(f'current patient_epochs:{patient_epochs}')
                        data_time = time.time()
                        iter_time = time.time()
                        train_data = prefetcher.next()
                    # end of iter

                    # update learning rate
                    model.update_learning_rate(
                            current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

                    epoch += 1
                    # validation each epoch
                    if opt.get('val') is not None and epoch>0:
                        rgb2bgr = opt['val'].get('rgb2bgr', True)
                        # wheather use uint8 image to compute metrics
                        use_image = opt['val'].get('use_image', True)
                        metric_results=model.validation(val_loader, current_iter, tb_logger,
                                        opt['val']['save_img'], rgb2bgr, use_image )
                        psnr_val_rgb=metric_results[0]
                        ssim_val_rgb=metric_results[1]
                        if ssim_val_rgb*10+psnr_val_rgb>best_score:
                            best_score=ssim_val_rgb*10+psnr_val_rgb
                            patient_epochs=0
                            best_name='BEST_PSNR_'+str(round(psnr_val_rgb,2))+'_SSIM_'+str(round(ssim_val_rgb,2))
                            model.save(epoch, best_name)
                            logger.info('current best psnr: %.4f and ssim : %.4f'%(psnr_val_rgb,ssim_val_rgb))
                        else:
                            patient_epochs+=1
                        logger.info(f'current patient_epochs:{patient_epochs}')
                    if epoch>total_epochs:
                        break
                    if patient_epochs>0 and patient_epochs%scale_t['patience']==0:
                        patient_epochs=0
                        break
    # end of epoch

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))  
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        model.validation(val_loader, current_iter, tb_logger,
                         opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()
