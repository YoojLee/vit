import argparse
from importlib import import_module
import numpy as np
import random
import os
import torch

def fix_seed(random_seed):
    """
    fix seed to control any randomness from a code 
    (enable stability of the experiments' results.)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def arg_parse():
    parser = argparse.ArgumentParser()

    # dataset.py 관련
    parser.add_argument("--data_root", type=str, default="ImageNet")
    parser.add_argument("--label_info", type=str, default="/home/data/label.txt")
    parser.add_argument("--dataset", type=str, default="ImageNetDataset")
    parser.add_argument("--transforms", type=str, default="BaseTransform")
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--downsample", action="store_true")

    # model.py 관련 하이퍼 파라미터
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--model_dim', type=int, default=768)
    parser.add_argument('--n_heads', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=3072)
    parser.add_argument('--n_class', type=int, default=1000)
    parser.add_argument('--p', type=int, default=16)
    parser.add_argument('--dropout_p', type=float, default=.1)
    parser.add_argument('--pool', type=str, default='cls')
    parser.add_argument('--drop_hidden', type=bool, default=False) # classification head에서 hidden layer drop

    # train.py 관련 하이퍼 파라미터
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--weight_decay', type=float, default=.3)
    parser.add_argument('--b1', type=float, default=.9)
    parser.add_argument('--b2', type=float, default=.999)
    parser.add_argument('--max_norm', type=int, default=1, help="max norm for gradient clipping")
    parser.add_argument('--accumulation_steps', type=int, default=32)

    # scheduler.py 관련 하이퍼 파라미터
    parser.add_argument('--lr_scheduler', type=str, default='WarmupCosineAnnealing')
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--period', type=int, default=-1)
    parser.add_argument('--warmup_restart', type=int, default=2000)
    parser.add_argument('--cycle_factor', type=float, default=1.0)
    parser.add_argument('--lr_verbose', type=bool, default=False)

    # miscellaneous
    parser.add_argument("--is_train", type=bool, default=True)
    parser.add_argument("--training_phase", type=str, default="p", help="p for pre-training or f for fine-tuning")
    parser.add_argument("--random_seed", type=int, default=0)

    # multi-processing
    parser.add_argument("--dist_backend", type=str, default='nccl')
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_id", type=int, nargs='+', default=-1) # if -1, use cpu
    parser.add_argument("--multi_gpu", type=bool, default=True)
    parser.add_argument("--n_nodes", type=int, default=1, help="number of cluster nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="a rank of the node")

    # wandb & logging
    parser.add_argument("--prj_name", type=str, default="vit")
    parser.add_argument("--exp_name", type=str, default="exp1")
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--sample_save_dir", type=str, default='test_results/') # only for test.py
    parser.add_argument("--last_checkpoint_dir", type=str, default="/home/workspace/weights/imagenet_cls_token/vit_98_0.66482.pt")
    parser.add_argument("--checkpoint_dir", type=str, default="weights")
    parser.add_argument("--resume_from", action="store_true")

    opt = parser.parse_args()

    return opt

# checkpoint
def save_checkpoint(checkpoint, saved_dir, file_name):
    os.makedirs(saved_dir, exist_ok=True) # make a directory to save a model if not exist.

    output_path = os.path.join(saved_dir, file_name)
    torch.save(checkpoint, output_path)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, rank=-1):
    # load model if resume_from is set
    
    if rank != -1: # distributed
        map_location = {"cuda:%d" % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    else:
        checkpoint = torch.load(checkpoint_path)    
        
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch']

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return model, optimizer, scheduler, start_epoch


def get_dataset(opt):
    # data loading
    dataset_module = import_module("dataset")
    aug_module = import_module("augmentation")
    augmentation_class = getattr(aug_module, opt.transforms)
    dataset_class = getattr(dataset_module, opt.dataset)

    augmentation = augmentation_class(opt.crop_size)
    train_data = dataset_class(opt.data_root, opt.p, opt.is_train, augmentation, opt.label_info, opt.downsample)
    val_data = dataset_class(opt.data_root, opt.p, not opt.is_train, augmentation, opt.label_info, opt.downsample)

    return train_data, val_data

def topk_accuracy(pred, true, k=1):
    pred_topk = pred.topk(k, dim=1)[1] # indices
    n_correct = torch.sum(pred_topk.squeeze() == true) # true: 크기가 b인 1차원 벡터

    return n_correct / len(true)

class AverageMeter(object):
    def __init__(self):
        self.init()
    
    def init(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count