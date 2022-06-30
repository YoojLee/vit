# reference: https://github.com/jeonsworld/ViT-pytorch/

import math
import torch
from torch.optim.lr_scheduler import LambdaLR

class WarmupLinearDecay(LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1, verbose=False):
        """
        Warm-up step 동안 linear warm-up을 적용하고 이후에 linear decay

        base_lr 같은 경우에는 optimizer에 저장이 되어 있음.

        어떤 식으로 linear decay 적용할지? base_lr부터 0까지 linearly decay되는 형식
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupLinearDecay, self).__init__(optimizer, self.lr_lambda, last_epoch, verbose) # self.lr_lambda를 메소드로 정의하기
    def lr_lambda(self, step):
        """
        base_lr 앞에 있는 계수를 리턴.
        """
        if step < self.warmup_steps: # warmup_steps 이전까지는 linear warmup이 적용됨.
            # 이게 왜 linear인지 생각해보자. lr_lambda는 step에 대한 함수임. 애초에 base_lr이 계수가 되는 구조임.
            # warmup_steps가 100이라면, lr은 .01*lr .02*lr .03*lr ... 이런 식으로 될 것임. 즉, .01씩 늘어나는 구조로 linear함.
            return float(step) / float(max(1, self.warmup_steps)) # 이게 왜 linear warmup이냐?
        return max(0.0, float(self.total_steps-step) / max(1.0, float(self.total_steps-self.warmup_steps)))# linear decay니까 어떤 식으로 linear하게 decay되어야 하는지 생각해보자.

class WarmupCosineDecay(LambdaLR):
    """
    Linear warmup and then cosine decay.

    cosine decay?

    """
    def __init__(self, optimizer, warmup_steps, total_steps, cycles=.5, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cycles = cycles
        super(WarmupCosineDecay, self).__init__(optimizer, self.lr_lambda, last_epoch, verbose)
    
    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        
        #progress = min(1, max(0,(step - self.warmup_steps) / float(self.total_steps - self.warmup_steps))) # clipping
        #return 0.5 * (1. + torch.cos(math.pi*progress)) # 이게 tensorflow 수식. 그래서 이거 어쩌란 거냐? vit-pytorch랑은 다른데..
        
        # progress after warmup (이 부분 나중에 수정할 것)
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))