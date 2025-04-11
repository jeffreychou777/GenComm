# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import statistics
from datetime import datetime
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

from icecream import ic
import wandb


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    opt = parser.parse_args()
    return opt


def main():
    
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    current_time = datetime.now()
    run_name = hypes['name'] + current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    wandb.init(project="DiffComm", name=run_name, config=hypes)
    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    if 'verify_mode' in hypes and hypes['verify_mode']:
        tiny_opencood_train_dataset = Subset(opencood_train_dataset, range(1300,2400))
        tiny_opencood_validate_dataset = Subset(opencood_validate_dataset, range(300))
        print("Verify mode, only use part samples")

        train_loader = DataLoader(tiny_opencood_train_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=4,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True,
                                prefetch_factor=2)
        val_loader = DataLoader(tiny_opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=4,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True,
                                prefetch_factor=2)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=4,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True,
                                prefetch_factor=2)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=4,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True,
                                prefetch_factor=2)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)
        print(f"resume from {init_epoch} epoch.")

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes, current_time)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        
    # record training
    writer = SummaryWriter(saved_path)

    print('Training start')
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    iter = 0
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        # the model will be evaluation mode during validation
        model.train()
        try: # heter_model stage2
            model.model_train_init()
        except:
            print("No model_train_init function")
        train_ave_loss = []
        for i, batch_data in enumerate(train_loader):
            iter += 1
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
                continue
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)

            batch_data['ego']['epoch'] = epoch
            ouput_dict = model(batch_data['ego'])
            
            # grad match loss
            loss_S, loss_T, gen_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])  #S -> gen, T -> org
            loss_S.backward(retain_graph=True)  # 计算 S 的梯度
            gradients_S = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
            optimizer.zero_grad()  # 清除梯度
            loss_T.backward(retain_graph=True)  # 计算 T 的梯度
            gradients_T = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
            grad_match_loss = sum(F.mse_loss(gradients_S[name], gradients_T[name]) for name in gradients_S)
            
            final_loss = loss_S + 0.1 * grad_match_loss + hypes['loss']['args']['generate_weight'] * gen_loss
            wandb.log({ 'train_loss_iter': final_loss,
                        'grad_match_loss': grad_match_loss,}, step=iter)
            
            train_ave_loss.append(final_loss.item())
            criterion.logging(epoch, i, len(train_loader), writer, iter=iter)

            if supervise_single_flag:
                final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
                criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")
            
            # back-propagation
            if final_loss.requires_grad:    ## in stage2, some case only contains ego,
                final_loss.backward()
            optimizer.step()
        train_ave_loss = statistics.mean(train_ave_loss)
        wandb.log({"epoch": epoch, "train_loss": train_ave_loss,}, step=iter)

            # torch.cuda.empty_cache()  # it will destroy memory buffer

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    ouput_dict = model(batch_data['ego'])

                    loss_S, _, gen_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    print(f'val loss {loss_S:.3f}')
                    val_loss = loss_S + gen_loss
                    valid_ave_loss.append(val_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)
            wandb.log({"epoch": epoch, "val_loss_epoch": valid_ave_loss,})

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    os.remove(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        scheduler.step(epoch)

        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)

    run_test = True
    if run_test:
        fusion_method = opt.fusion_method
        cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
        print(f"Running command: {cmd}")
        os.system(cmd)

if __name__ == '__main__':
    main()
