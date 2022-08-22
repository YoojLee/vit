# reference: https://github.com/jeonsworld/ViT-pytorch/

import math
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
    """
    def __init__(self, optimizer, warmup_steps, total_steps, cycle_factor=1., last_epoch=-1, verbose=False):
        """
        - Args
            optimizer: optimizer to adjust learning rate
            warmup_steps: steps for a learning rate warm-ups (linear warmup)
            total_steps: total steps
            cycle_factor: factor for adjusting a period T_i
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cycle_factor = cycle_factor # 주기에 적용하는 factor인 듯함.
        super(WarmupCosineDecay, self).__init__(optimizer, self.lr_lambda, last_epoch, verbose)
    
    def lr_lambda(self, step):
        # warm-up 적용
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        
        # min_eta = 0, max_eta = initial_lr로 가정. (그렇게 되니까 lr_lambda를 그냥 쓰는 거겠지 -> lr_lambda 자체가 lambda function으로 초기 학습률값을 조작하는 방식으로 동작)
        # 얘는 annealing이 아니라 그냥 전체 step을 하나의 주기로 보는 것 같음. annealing을 하나 더 구현해보자!
        progress = min(1, max(0,(step - self.warmup_steps) / float(self.total_steps - self.warmup_steps))) # clipping (progress를 0과 1 사이로 강제)
        return 0.5 * (1. + math.cos(math.pi*progress*float(self.cycle_factor)))


class WarmupCosineAnnealing(LambdaLR):
    """
    Cosine Annealing with Warm Restarts after Linear Warm-Up.
    For Cosine Annealing, minimum learning rate implicitly set to zero and maximum learning rate implicitly set to the initial learning rate.
    """
    def __init__(self, optimizer, warmup_steps, period, warmup_restart, cycle_factor=1, ceil_lr_factor=0, floor_lr_factor=1, last_epoch=-1, verbose=False):
        """
        - Args
            optimizer: optimizer to adjust a learning rate
            warmup_steps: number of steps for warm up (it occurs only once at the beginning of training)
            period: period for cosine annealing
            warmup_restart: if larger than zero, the learning rate is reset to its initial state with linear warm-up, not an immediate manner.
            cycle_factor: a factor for adjusting a length of the period. (default=.2)
            last_epoch: at which epoch(or step) the training progress is (default=-1)
        """
        self.warmup_steps = warmup_steps
        self.t_cur = 0
        self.period = period
        self.warmup_restart = warmup_restart
        self.cycle_factor = float(cycle_factor)
        self.ceil_lr_factor = float(ceil_lr_factor)
        self.floor_lr_factor = float(floor_lr_factor)
        super(WarmupCosineAnnealing, self).__init__(optimizer, self.lr_lambda, last_epoch, verbose)

    def lr_lambda(self, step):
        # warm-up
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))

        # cosine annealing
        if (step // self.warmup_steps == 1):
            progress = min(self.floor_lr_factor, max(self.ceil_lr_factor, (self.t_cur) / (self.period))) # clipping to range [0,1]
        else:
            if (self.t_cur < self.warmup_restart):
            # warm restarts
                self.t_cur += 1
                return float(self.t_cur-1) / float(max(1, self.warmup_restart))
            else:
                progress =  min(self.floor_lr_factor, max(self.ceil_lr_factor, (self.t_cur-self.warmup_restart) / (self.period-self.warmup_restart)))
                    
        if self.t_cur == self.period: # 한 주기가 끝나면
            self.period = int(self.period*(1+self.cycle_factor)) # 여기 int로 안바꿔주면, 주기가 소수가 되기 때문에 t_cur와의 비교가 어려움.
            self.t_cur = 0
        else:
            self.t_cur += 1
        
        return 0.5 * (1+math.cos(progress*math.pi))