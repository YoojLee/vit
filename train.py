from importlib import import_module

from model import *
from utils import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import tqdm
import wandb

def validate(val_loader, model, device):
    model.eval()

    print("Validation Starts")

    criterion = nn.CrossEntropyLoss()
    classes = val_loader.dataset.label_names

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    n_correct, total = 0,0

    pbar = tqdm.tqdm(enumerate(val_loader), total=len(val_loader))

    with torch.no_grad():
        for step, (patches, labels) in pbar:
            patches = patches.to(device)
            labels = labels.to(device)
            
            outputs = model(patches)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            for label, pred in zip(labels, preds):
                if label == pred:
                    correct_pred[classes[label-1]] += 1
                    n_correct += 1
                total_pred[classes[label-1]] += 1
                total += 1

            description = f'Validation Step: {step+1}/{len(val_loader)} || Validation Loss: {round(loss.item(), 4)} || Validation Accuracy: {round(n_correct/total, 4)}'
            pbar.set_description(description)
    
            # wandb logging
            wandb.log(
                {   
                    'Validation Loss': round(loss.item(), 4),
                    'Validation Accuracy': round(n_correct/total, 4)
                }
            )



def train(train_loader, val_loader, opt, device, total_len):
    n_patches = int(opt.crop_size**2 / opt.p**2)

    # model
    model = ViT(opt.p, opt.model_dim, opt.hidden_dim, opt.n_class, opt.n_heads, opt.n_layers, n_patches, opt.dropout_p, opt.training_phase, opt.pool)
    model = model.to(device)
    
    # optimizer
    optimizer_class = getattr(import_module("torch.optim"), opt.optimizer)
    optimizer = optimizer_class(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

    # loss
    criterion = nn.CrossEntropyLoss()

    scheduler_class = getattr(import_module("scheduler"), opt.lr_scheduler)
    scheduler = scheduler_class(optimizer, opt.warmup_steps, total_len*opt.n_epochs, verbose=False) # ?????? scheduler??? step??? ??????????????? ?????? ?????????. 10k step?????? epoch??? ?????? ??????????????? step looping??? ??? iteration??? ????????? ???????????
    # total steps ?????? ???????????? batch size??? dependent. (?????? ????????? ?????? / batch size * num_epochs ?????? total steps??? ????????? ???!)


    # training loop
    for epoch in range(1, opt.n_epochs+1):
        
        model.train() # ??? epoch?????? ?????? validation??? ????????? ???. ?????????, ??? epoch?????? train ????????? ?????? ??????????????? ???.
        running_loss = 0.0
        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))

        for step, (patches, labels) in pbar:
            # gradients flushing
            optimizer.zero_grad()
            
            # device??? ????????????
            patches = patches.to(device) # patch??? ???????????? ??????????????? ??? ?????????.
            labels = labels.to(device)

            # forward
            y_pred = model(patches)
            loss = criterion(y_pred, labels) # The input is expected to contain raw, unnormalized scores for each class
            running_loss += loss.item() # loss??? ?????? loss.mean??? ?????? ?????????.
            # backward
            loss.backward()

            # gradient clipping
            if opt.dataset == "ImageNetDataset":
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.max_norm)
            
            # optimize
            optimizer.step()

            # logging
            description = f'Epoch: {epoch}/{opt.n_epochs} || Step: {step+1}/{len(train_loader)} || Training Loss: {round(loss.item(), 4)}'
            pbar.set_description(description)

            if (opt.log_interval > 0) and ((step+1) % opt.log_interval == 0):
                wandb.log(
                    {
                        "Training Loss": round(running_loss/opt.log_interval, 4)
                    }
                )
                running_loss = 0.0

            # scheduler -> scheduler??? ????????? ????????? step??? ????????? ?????????
            scheduler.step()
            wandb.log(
                {
                    "Learning Rate": optimizer.param_groups[0]['lr']
                }
            )
        # after every epoch, Run validate
        validate(val_loader, model, device)



def main():
    # parse arguments
    opt = arg_parse()

    # fix seed
    fix_seed(opt.random_seed)

    # wandb logging initialization
    wandb.init(project=opt.prj_name, name=opt.exp_name, entity="yoojlee", config=vars(opt))

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data loading
    dataset_module = import_module("dataset")
    aug_module = import_module("augmentation")
    augmentation_class = getattr(aug_module, opt.transforms)
    dataset_class = getattr(dataset_module, opt.dataset)

    augmentation = augmentation_class(opt.resize, opt.crop_size)
    train_data = dataset_class(opt.data_root, opt.p, opt.is_train, augmentation, opt.label_info, opt.downsample)
    val_data = dataset_class(opt.data_root, opt.p, not opt.is_train, augmentation, opt.label_info, opt.downsample)

    # for dataloader reproducibility
    g = torch.Generator()
    g.manual_seed(opt.random_seed)
    
    shuffle=True
    train_sampler = None
    val_sampler = None
    
    # ?????? ??? ???????????? ??????????????? trainer ????????? ?????? ?????? ????????? ??????...
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=shuffle, num_workers=opt.num_workers, worker_init_fn=seed_worker, generator=g, sampler=train_sampler, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=shuffle, num_workers=opt.num_workers, worker_init_fn=seed_worker, generator=g, sampler=val_sampler, pin_memory=True)


    # train
    train(train_loader, val_loader, opt, device, total_len=len(train_data))

    # close wandb session
    wandb.run.finish()

    
if __name__ == "__main__":
    main()
