from importlib import import_module

from model import *
from utils import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

from datetime import timedelta
from tqdm import tqdm
import wandb

def main():
    """
    argument parsing 해와서 multiprocessing spawning해주기 (하나의 머신에서만 동작한다고 가정함)
    """
    opt = arg_parse()
    fix_seed(opt.random_seed)

    ngpus_per_node = len(opt.gpu_id)
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
    
    
    dist.init_process_group(
        backend=opt.dist_backend, # nccl: gpu 백엔드
        init_method=f'tcp://127.0.0.1:11203',
        world_size=opt.world_size,
        rank=opt.rank, # rank가 gpu가 아닌가?
        timeout=timedelta(300)
    )


    # 모델 정의
    print(f"==> Creating ViT Instance...({opt.dist_backend})")

    n_patches = int(opt.crop_size**2 / opt.p**2)
    model = ViT(opt.p, opt.model_dim, opt.hidden_dim, opt.n_class, opt.n_heads, opt.n_layers, n_patches, opt.dropout_p, opt.training_phase, opt.pool, opt.drop_hidden)
    model.cuda(opt.gpu)
    batch_size = int(opt.batch_size / ngpus_per_node)
    num_workers = int(opt.num_workers / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu])

    # wandb init
    if opt.rank == 0:
        wandb.init(project=opt.prj_name, name=f"{opt.exp_name}", entity="yoojlee", config=vars(opt))
        wandb.watch(model, log='all', log_freq=opt.log_interval)

    criterion = nn.CrossEntropyLoss(reduction='mean').cuda(opt.gpu)
    
    # optimizer
    optimizer_class = getattr(import_module("torch.optim"), opt.optimizer)
    optimizer = optimizer_class(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

    # scaler
    scaler = GradScaler() 

    # dataset
    train_dataset, val_dataset = get_dataset(opt) # 여기가 하나의 병목

    train_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=(train_sampler is None), 
                              num_workers=num_workers, 
                              pin_memory=True,
                              sampler=train_sampler,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # scheduler
    if opt.period == -1:
        opt.period = int((len(train_dataset) / (opt.batch_size*opt.accumulation_steps)) * opt.n_epochs)
        if opt.rank == 0:
            wandb.config.update({"period":opt.period}, allow_val_change=True)

    scheduler_class = getattr(import_module("scheduler"), opt.lr_scheduler)
    scheduler = scheduler_class(optimizer, opt.warmup_steps, opt.period, opt.warmup_restart, cycle_factor=opt.cycle_factor, verbose=opt.lr_verbose) # annealing에 period = total_step으로 넣어주면 cosine decay로 적용될 듯.


    # resume from
    if opt.resume_from:
        model, optimizer, scheduler, start_epoch = load_checkpoint(opt.last_checkpoint_dir, model, optimizer, scheduler, opt.rank)
        if scheduler.__dict__['cycle_factor'] != opt.cycle_factor:
            print("Updating cycle factor of scheduler")
            scheduler.__dict__['cycle_factor'] = opt.cycle_factor

    else:
        start_epoch = 0

    dist.barrier()
    for epoch in range(start_epoch, opt.n_epochs):
        train_sampler.set_epoch(epoch) # sampler가 동일한 data를 생성하는 것을 방지하기 위해

        # reset gradients
        optimizer.zero_grad()

        _ = train(train_loader, model, criterion, optimizer, scheduler, scaler, epoch, opt) # model.module과 model의 차이?
        
        if (opt.rank==0):
            acc_score, _ = validate(val_loader, model, criterion, epoch, opt)

            if (best_top1 < acc_score):
                best_top1 = acc_score
                print(f"Saving Weights at Accuracy {best_top1}")
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'best_top1': best_top1,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }, os.path.join(opt.checkpoint_dir, opt.exp_name), f"vit_{epoch}_{best_top1}.pt" # 이 부분이 문제였음. 애초에 error가 발생해도 프로그램이 멈추는 게 아니라 그냥 계속 좀비 프로세스처럼 남아있는 느낌..?
                )
            
                print(f"Best Accuracy: {best_top1}")

        torch.cuda.empty_cache()
    
    if opt.rank==0:
        wandb.run.finish()

    dist.destroy_process_group()


def train(train_loader, model, criterion, optimizer, scheduler, scaler, epoch, opt): # 하나의 epoch 내에서 동작하는 함수
    model.train()
    acc_score, running_loss = AverageMeter(), AverageMeter() # 이 부분을 log_interval에서의 average metric으로 사용할 수 있도록 바꿔놔야할 것 같음.
    pbar = tqdm(enumerate(train_loader), total=len(train_loader)) # 왜인지는 모르겠는데 여기도 하나의 병목

    for step, (image, labels) in pbar: # 애초에 여기서 generating이 안됨.
        # device에 올려주기
        image = image.cuda()
        labels = labels.cuda()
        
        # forward
        with autocast():
            y_pred = model(image)
            loss = criterion(y_pred, labels) # The input is expected to contain raw, unnormalized scores for each class
            loss = loss / opt.accumulation_steps
        
        scaler.scale(loss).backward()

        running_loss.update(loss.item()*opt.accumulation_steps,  image.size(0))
        acc_score.update(topk_accuracy(y_pred.clone().detach(), labels).item(), image.size(0))
        
        if (step+1) % opt.accumulation_steps == 0:

            # gradient clipping
            if opt.dataset == "ImageNetDataset":
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.max_norm)

            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            # scheduler step
            scheduler.step()
            
            # gradient reset
            optimizer.zero_grad()

            # logging
            dist.barrier()
            if (opt.rank == 0) and ((step+1) % opt.log_interval == 0):
                wandb.log(
                        {
                            "Training Loss": round(running_loss.avg, 4),
                            "Training Accuracy": round(acc_score.avg, 4),
                            # "Learning Rate (from scheduler)": scheduler.get_last_lr()[0]
                            "Learning Rate": optimizer.param_groups[0]['lr']
                        }
                )
                running_loss.init()
                acc_score.init()

        description = f'Epoch: {epoch+1}/{opt.n_epochs} || Step: {(step+1)//opt.accumulation_steps}/{len(train_loader)//opt.accumulation_steps} || Training Loss: {round(running_loss.avg, 4)}'
        pbar.set_description(description)

    return running_loss.avg

def validate(val_loader, model, criterion, epoch, opt): 
    model.eval()

    losses = AverageMeter()
    acc_score = AverageMeter()
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    
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
    dist.barrier()
    if opt.rank == 0:
        wandb.log(
                {
                    'Validation Loss': round(losses.avg, 4),
                    'Validation Accuracy': round(acc_score.avg, 4)
                }
            )

    return acc_score.avg, losses.avg

            

if __name__ == "__main__":
    os.environ['NCCL_DEBUG']='INFO'
    main()