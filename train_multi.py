from importlib import import_module

from model import *
from utils import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import builtins
from datetime import timedelta
import os
import tqdm
import wandb

def main():
    """
    argument parsing 해와서 multiprocessing spawning해주기 (하나의 머신에서만 동작한다고 가정함)
    """
    opt = arg_parse()
    opt.gpu_id = [0,1]
    opt.world_size = 1 # 머신 개수
    opt.downsample = True
    opt.batch_size = 128
    opt.rank = 0

    ngpus_per_node = torch.cuda.device_count()
    opt.world_size = ngpus_per_node * opt.world_size
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt)) # process 뿌려주기
    

def main_worker(gpu, ngpus_per_node, opt):
    """
    각 process가 수행하는 함수.

    main_worker 내부에서 training loop이 구성이 됨.
    """
    global best_top1
    best_top1 = 0.0

    opt.gpu = gpu
    torch.cuda.set_device(opt.gpu)

    print(f"==> Device {opt.gpu} working...")
    opt.rank = opt.rank * ngpus_per_node + opt.gpu # 원래의 opt.rank는 local rank (해당 노드에서 몇번째 프로세스인지)를 의미하는 듯함.
    
    
    # 각 process 별로 병렬 수행 중, print가 프로세스 별로 중복되는 것을 방지하기 위해서 0번 process를 제외하고는 print를 하지 않는다.
    # if opt.multi_gpu and opt.rank != 0:
    #     def print_pass(*args):
    #         pass
    #     builtins.print = print_pass
    
    dist.init_process_group(
        backend=opt.dist_backend, # nccl: gpu 백엔드
        init_method=f'tcp://127.0.0.1:11203',
        world_size=opt.world_size,
        rank=opt.rank,
        timeout=timedelta(300)
    )

    if opt.rank == 0:
        dist.barrier()
        wandb.init(project=opt.prj_name, name=f"{opt.exp_name}", entity="yoojlee", config=vars(opt))

    # 모델 정의
    print(f"==> Creating ViT Instance...({opt.dist_backend})")

    n_patches = int(opt.crop_size**2 / opt.p**2)
    model = ViT(opt.p, opt.model_dim, opt.hidden_dim, opt.n_class, opt.n_heads, opt.n_layers, n_patches, opt.dropout_p, opt.training_phase, opt.pool)
    model.cuda(opt.gpu)
    batch_size = int(opt.batch_size / ngpus_per_node)
    num_workers = int(opt.num_workers / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu])

    criterion = nn.CrossEntropyLoss().cuda(opt.gpu)
    
    # optimizer
    optimizer_class = getattr(import_module("torch.optim"), opt.optimizer)
    optimizer = optimizer_class(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

    # dataset
    train_dataset, val_dataset = get_dataset(opt) # 여기가 하나의 병목

    train_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=(train_sampler is None), 
                              num_workers=num_workers, 
                              pin_memory=True, 
                              sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    # scheduler
    scheduler_class = getattr(import_module("scheduler"), opt.lr_scheduler)
    scheduler = scheduler_class(optimizer, opt.warmup_steps, len(train_dataset)*opt.n_epochs, verbose=False)

    # resume from
    if opt.resume_from:
        model, optimizer, scheduler, start_epoch = load_checkpoint(opt.checkpoint_dir, model, optimizer, scheduler)

    else:
        start_epoch = 0

    for epoch in range(start_epoch, opt.n_epochs):
        train_sampler.set_epoch(epoch) # sampler가 동일한 data를 생성하는 것을 방지하기 위해

        _ = train(train_loader, model.module, criterion, optimizer, epoch, opt)
        acc_score, _ = validate(val_loader, model.module, criterion, epoch, opt)

        best_top1 = max(acc_score, best_top1)

        # scheduler step
        scheduler.step()
        
        if not opt.multi_gpu or (opt.multi_gpu and opt.rank==0):
            dist.barrier()
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
    
    # if opt.rank==0:
    #     wandb.run.finish()

    dist.destroy_process_group()


def train(train_loader, model, criterion, optimizer, epoch, opt): # 하나의 epoch 내에서 동작하는 함수
    model.train()
    acc_score, running_loss = AverageMeter(), AverageMeter()
    pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader)) # 왜인지는 모르겠는데 여기도 하나의 병목

    for step, (patches, labels) in pbar: # 애초에 여기서 generating이 안됨.
        # device에 올려주기
        patches = patches.cuda()
        labels = labels.cuda()

        # forward
        y_pred = model(patches)
        loss = criterion(y_pred, labels) # The input is expected to contain raw, unnormalized scores for each class

        running_loss.update(loss.item(),  patches.size(0))
        acc_score.update(topk_accuracy(y_pred.detach(), labels).item(), patches.size(0))

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

        # if (opt.rank == 0) and (opt.log_interval > 0) and ((step+1) % opt.log_interval == 0):
        #     wandb.log(
        #             {
        #                 "Training Loss": round(running_loss.avg, 4),
        #                 "Training Accuracy": round(acc_score.avg, 4)
        #             }
        #     )

    return running_loss.avg

def validate(val_loader, model, criterion, epoch, opt):
    model.eval()

    losses = AverageMeter()
    acc_score = AverageMeter()
    pbar = tqdm.tqdm(enumerate(val_loader), total=len(val_loader))
    
    with torch.no_grad():
        for step, (patches, labels) in pbar:
            patches = patches.cuda()
            labels = labels.cuda()
            
            y_pred = model(patches)
            loss = criterion(y_pred, labels)

            losses.update(loss.item(), patches.size(0))
            acc_score.update(topk_accuracy(y_pred.detach(), labels).item(), patches.size(0))

            description = f'Current Epoch: {epoch+1} || Validation Step: {step+1}/{len(val_loader)} || Validation Loss: {round(loss.item(), 4)} || Validation Accuracy: {round(acc_score.avg, 4)}'
            pbar.set_description(description)

     # rank == 0 으로 지정해주지 않으면 wandb init을 시행하지 않은 것으로 간주함.
    # if opt.rank == 0:
    #     wandb.log(
    #             {
    #                 'Validation Loss': round(losses.avg, 4),
    #                 'Validation Accuracy': round(acc_score.avg, 4)
    #             }
    #         )

    return acc_score.avg, losses.avg

            

if __name__ == "__main__":
    #torch.multiprocessing.set_start_method('forkserver')
    main()