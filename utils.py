import argparse
import torch
import random
import numpy as np


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
    parser.add_argument("--label_info", type=str, default="label.txt")
    parser.add_argument("--dataset", type=str, default="ImageNetDataset")
    parser.add_argument("--transforms", type=str, default="BaseTransform")
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    
    # model.py 관련 하이퍼 파라미터
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--model_dim', type=int, default=768)
    parser.add_argument('--n_heads', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=3072)
    parser.add_argument('--n_class', type=int, default=1000)
    parser.add_argument('--p', type=int, default=16)
    parser.add_argument('--dropout_p', type=float, default=.1)

    # train.py 관련 하이퍼 파라미터
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--weight_decay', type=float, default=.3)
    parser.add_argument('--b1', type=float, default=.9)
    parser.add_argument('--b2', type=float, default=.999)
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR')
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--max_norm', type=int, default=1, help="max norm for gradient clipping")

    # miscellaneous
    parser.add_argument("--is_train", type=bool, default=True)
    parser.add_argument("--training_phase", type=str, default="p", help="p for pre-training or f for fine-tuning")
    parser.add_argument("--num_threads", type=int, default=24)
    parser.add_argument("--gpu_id", type=str, default="0,1")
    parser.add_argument("--random_seed", type=int, default=0)

    # wandb & logging
    parser.add_argument("--prj_name", type=str, default="vit")
    parser.add_argument("--exp_name", type=str, default="exp1")
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--sample_save_dir", type=str, default='test_results/') # only for test.py
    parser.add_argument("--last_checkpoint_dir", type=str, default="weights/exp1")
    parser.add_argument("--checkpoint_dir", type=str, default="weights")
    parser.add_argument("--load_epoch", type=int, default=150)
    parser.add_argument("--resume_from", action="store_true")

    opt = parser.parse_args()

    return opt
    
    



