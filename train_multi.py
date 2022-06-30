from importlib import import_module

from sklearn.metrics import accuracy_score

from model import *
from utils import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import builtins
import os
import tqdm
import wandb

def main():
    """
    argument parsing 해와서 multiprocessing spawning해주기 (하나의 머신에서만 동작한다고 가정함)
    """
    opt = arg_parse()
    opt.gpu_id = [0,1]
    opt.world_size = len(opt.gpu_id)

    wandb.init(project=opt.prj_name, name=opt.exp_name, entity="yoojlee", config=vars(opt))

    if opt.world_size > 1 or opt.multi_gpu: # multi-processing applied
        torch.multiprocessing.spawn(main_worker, nprocs=opt.world_size, args=(opt.world_size, opt)) # process 뿌려주기
    else:
        main_worker(opt.gpu_id, opt.world_size, opt)

    wandb.run.finish()
    

def main_worker(rank, n_gpus, opt):
    """
    각 process가 수행하는 함수.

    main_worker 내부에서 training loop이 구성이 됨.
    """
    opt.rank = rank

    global best_top1
    
    if rank != -1:
        print(f"==> Device {rank} working...")

    # 각 process 별로 병렬 수행 중, print가 프로세스 별로 중복되는 것을 방지하기 위해서 0번 process를 제외하고는 print를 하지 않는다.
    if opt.multi_gpu and rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    
    if opt.multi_gpu: # multi-gpu training
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:53132',
            world_size=n_gpus,
            rank=rank # rank가 gpu가 아닌가?
        )

    # 모델 정의
    print("==> Creating ViT Instance...")

    n_patches = int(opt.crop_size**2 / opt.p**2)
    model = ViT(opt.p, opt.model_dim, opt.hidden_dim, opt.n_class, opt.n_heads, opt.n_layers, n_patches, opt.dropout_p, opt.training_phase, opt.pool)

    # multi-process 설정
    if opt.multi_gpu:
        if opt.gpu_id != -1: # gpu_id가 제대로 할당이 되어 있다면
            torch.cuda.set_device(rank)
            model.to(rank)

            batch_size = int(opt.batch_size / n_gpus)
            num_workers = int(opt.num_workers / n_gpus)

            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

        else: # gpu_id가 제대로 할당이 되어 있지 않다면
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)

    # single_gpu
    elif rank != -1:
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        model.to(device)

    criterion = nn.CrossEntropyLoss().cuda(rank)
    
    # optimizer
    optimizer_class = getattr(import_module("torch.optim"), opt.optimizer)
    optimizer = optimizer_class(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

    # dataset
    train_dataset, val_dataset = get_dataset(opt) # 여기가 하나의 병목

    if opt.multi_gpu:
        train_sampler = DistributedSampler(train_dataset, num_replicas=n_gpus, rank=rank, drop_last=True)
    else:
        train_sampler = None
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=(train_sampler is None), 
                              num_workers=num_workers, 
                              pin_memory=True, 
                              sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # scheduler
    scheduler_class = getattr(import_module("scheduler"), opt.lr_scheduler)
    scheduler = scheduler_class(optimizer, opt.warmup_steps, len(train_dataset)*opt.n_epochs, verbose=False)

    # resume from
    # dist.barrier() # 병목
    if opt.resume_from:
        model, optimizer, scheduler, start_epoch = load_checkpoint(opt.checkpoint_dir, model, optimizer, scheduler)

    else:
        start_epoch = 0

    for epoch in range(start_epoch, opt.n_epochs):
        if opt.multi_gpu:
            train_sampler.set_epoch(epoch)

        train_loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        acc_score, val_loss = validate(val_loader, model, criterion, epoch, opt)

        best_top1 = min(acc_score, best_top1)

        # scheduler step
        scheduler.step()

        if not opt.multi_gpu or (opt.multi_gpu and opt.rank == 0):
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_top1': best_top1,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, opt.saved_dir, f"vit_{epoch}_{best_top1}.pt"
            )
        
        print(f"Best Accuracy: {best_top1}")


def train(train_loader, model, criterion, optimizer, epoch, opt): # 하나의 epoch 내에서 동작하는 함수
    model.train()
    acc_score, running_loss = AverageMeter(), AverageMeter()
    pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader)) # 왜인지는 모르겠는데 여기도 하나의 병목

    dist.barrier()
    for step, (patches, labels) in pbar: # 애초에 여기서 generating이 안됨.
        # device에 올려주기
        patches = patches.cuda()
        labels = labels.cuda()

        # forward
        y_pred = model(patches)
        loss = criterion(y_pred, labels) # The input is expected to contain raw, unnormalized scores for each class

        running_loss.update(loss.item(),  patches.size(0))
        acc_score.update(accuracy_score(y_pred.detach(), labels).item(), patches.size(0))

        # backward
        optimizer.zero_grad() # 이게 병목인가? 왜 안되지?
        loss.backward()

        # gradient clipping
        if opt.dataset == "ImageNetDataset":
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.max_norm)
            
        # optimize
        optimizer.step()

        # logging
        description = f'Epoch: {epoch+1}/{opt.n_epochs} || Step: {step+1}/{len(train_loader)} || Training Loss: {round(loss.item(), 4)}'
        pbar.set_description(description)

        if (opt.log_interval > 0) and ((step+1) % opt.log_interval == 0):
            wandb.log(
                    {
                        "Training Loss": round(running_loss.avg, 4),
                        "Training Accuracy": round(acc_score.avg, 4)
                    }
            )

    return running_loss.avg

def validate(val_loader, model, criterion, epoch, opt):
    losses = AverageMeter()
    acc_score = AverageMeter()
    pbar = tqdm.tqdm(enumerate(val_loader), total=len(val_loader))
    
    with torch.no_grad():
        for step, (patches, labels) in pbar:
            patches = patches.cuda()
            labels = labels.cuda()
            
            y_pred = model(labels)
            loss = criterion(y_pred, labels)

            acc_score = accuracy_score(y_pred.detach(), labels)

            losses.update(loss.item(), patches.size(0))
            acc_score.update(acc_score.item(), patches.size(0))

            description = f'Validation Step: {step+1}/{len(val_loader)} || Validation Loss: {round(loss.item(), 4)} || Validation Accuracy: {round(acc_score, 4)}'
            pbar.set_description(description)

        wandb.log(
            {
                'Validation Loss': round(losses.avg, 4),
                'Validation Accuracy': round(acc_score.avg, 4)
            }
        )

    return acc_score.avg, losses.avg

            

if __name__ == "__main__":
    main()