import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Embeddings(nn.Module):
    def __init__(self, p:int, input_dim: int, model_dim:int, n_patches:int, dropout_p:float):
        """
        patch embedding과 positional embedding 생성하는 부분 (Encoder 인풋 만들어주는 부분이라고 생각하면 됨)
        """
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.n_patches = n_patches

        self.to_patch_embedding = nn.Sequential(
            OrderedDict(
                {
                    "rearrange": Rearrange('b c (h1 p1) (w1 p2) -> b (h1 w1) (p1 p2 c)', p1 = p, p2 = p), # h1 = h / p1, w1 = w / p2
                    "projection": nn.Linear(self.input_dim, self.model_dim)
                })
        )

        # 여기서는 cls_token을 정의해주면 됨. 어떤 식으로 initialization할지가 문제 -> 이거 이렇게 그냥 randn으로 해줘도 되나?
        self.cls_token = nn.Parameter(torch.randn(1,1,model_dim)) # 바로 이런 식으로 하면 안될라나?... 1로 해준 다음에 repeat해줘야 하나?

        # self.position_embedding
        self.pos_emb = nn.Parameter(torch.randn(1,n_patches+1,model_dim))

        # dropout
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x):
        """
        x: an image tensor
        """
        proj = self.to_patch_embedding(x)
        b, _, _ = proj.shape

        # embedding
        # cls_token = self.cls_token.repeat(b,1,1) # 이렇게 해야 b,1,model_dim으로 됨.
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)

        # summation
        patch_emb = torch.cat((cls_token, proj), dim = 1)
        
        return self.dropout(self.pos_emb + patch_emb)


class MSA(nn.Module):
    def __init__(self, model_dim:int, n_heads:int, dropout_p:float, drop_hidden:bool):
        """
        Multi-Head Self-Attention Block.

        - Args
            model_dim (int): model dimension D
            n_heads (int): number of heads
            dropout_p (float): a probability of a dropout masking
        """
        super().__init__()

        self.model_dim = model_dim
        self.n_heads = n_heads
        self.dropout_p = dropout_p
        self.drop_hidden = drop_hidden
        self.scale = (model_dim//n_heads) ** -0.5

        self.norm = nn.LayerNorm(model_dim) # LayerNorm은 입력 배치 내에서 통계량을 계산하는 것.
        self.attend = nn.Softmax(dim=-1)
        self.linear_qkv = nn.Linear(model_dim, 3*model_dim, bias=False)# [U_qkv] for changing the dimension 
        self.projection = nn.Identity() if self.drop_hidden else nn.Sequential(nn.Linear(model_dim, model_dim), nn.Dropout(dropout_p))

    
    def forward(self, z):
        b,n,_ = z.shape
        # [B, N, 3*D] -> 3개의 [B,k,N,D_h]로 쪼개는 게 목표
        qkv = self.linear_qkv(self.norm(z)).chunk(3, dim=-1) # chunk method는 tuple을 리턴함. ([B,N,D],[B,N,D],[B,N,D])

        # destack qkv into q,k,v (3 vectors)
        # qkv_destack = qkv.reshape(b,n,3,self.n_heads,-1) # 이렇게 해줘야 b,n 차원은 건드리지 않고 벡터 차원만 가지고 크기 조작이 가능함. 마지막 차원은 d_h와 동일함.
        q,k,v = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h = self.n_heads), qkv) # [b,n,1,k,d_h] 차원의 벡터 3개를 리턴
        
        # q = q.squeeze().transpose(1,2) # 이 방식은 batch size가 1일 때는 문제가 될 것 같다.
        # k = k.squeeze().transpose(1,2)
        # v = v.squeeze().transpose(1,2)

        # q, k attention
        qk_T = torch.matmul(q,k.mT)*self.scale # [b,k,n,n]
    
        # attention = F.softmax(qk_T*self.scale, dim=-1)
        attention = self.attend(qk_T)

        if self.dropout_p:
            attention = F.dropout(attention, p=self.dropout_p)

        # compute a weighted sum of v
        # msa = torch.matmul(attention, v).transpose(1,2)
        msa = torch.matmul(attention, v)
        msa_cat = rearrange(msa, 'b h n d -> b n (h d)')

        # concatenate k attention heads
        # msa_cat = msa.reshape(b,n,self.model_dim)

        # projection
        output = self.projection(msa_cat)

        # dropout
        # if self.dropout_p:
        #     output = F.dropout(output, p=self.dropout_p)
        
        return z+output # skip-connection
    

class FFN(nn.Module):
    def __init__(self, model_dim, hidden_dim, dropout_p):
        super().__init__()

        self.model_dim = model_dim
        self.hidden_dim = hidden_dim

        self.norm = nn.LayerNorm(self.model_dim)
        self.fc1 = nn.Linear(self.model_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.model_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)

        self.block = nn.Sequential(
            self.norm,
            self.fc1,
            self.activation,
            self.dropout,
            self.fc2,
            self.dropout # 여기에는 gelu를 쓰지 X.
        )

    def forward(self, z):
        return z + self.block(z)


class Encoder(nn.Module):
    def __init__(self, n_layers, model_dim, n_heads, hidden_dim, dropout_p, drop_hidden):
        super().__init__()

        self.n_layers = n_layers
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.drop_hidden = drop_hidden
        
        layers = []

        for _ in range(self.n_layers):
            layers.append(nn.Sequential(
                MSA(self.model_dim, self.n_heads, self.dropout_p, self.drop_hidden),
                FFN(self.model_dim, self.hidden_dim, self.dropout_p)
            )) # 반복해서 넣어줘야 하는 레이어들은 속성으로 할당해서 쓰면 안됨. 그때마다 객체를 새롭게 정의해야 함!
        
        self.encoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.encoder(z)


class ClassificationHead(nn.Module):
    def __init__(self, model_dim, n_class, training_phase, dropout_p, pool:str):
        super().__init__()

        self.model_dim = model_dim
        self.n_class = n_class
        self.training_phase = training_phase

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.pool = pool

        self.norm = nn.LayerNorm(self.model_dim)
        self.hidden = nn.Linear(self.model_dim, self.n_class)
        self.dropout = nn.Dropout(dropout_p)
        self.relu = nn.ReLU(inplace=True)

        self.block = nn.Sequential(self.norm, self.hidden)
    
    def forward(self, encoder_output):
        y = encoder_output.mean(dim=1) if self.pool == 'mean' else encoder_output[:, 0] # cls_token으로 predict할 경우 첫번째 요소만 slicing

        return self.block(y) # pre-norm 적용
        

class ViT(nn.Module):
    def __init__(self, p, model_dim, hidden_dim, n_class, n_heads, n_layers, n_patches, dropout_p=.1, training_phase='p', pool='cls', drop_hidden=True):
        super().__init__()
        input_dim = (p**2)*3
        
        self.vit = nn.Sequential(
            OrderedDict({
                "embedding": Embeddings(p, input_dim, model_dim, n_patches, dropout_p),
                "encoder": Encoder(n_layers, model_dim, n_heads, hidden_dim, dropout_p, drop_hidden),
                "c_head": ClassificationHead(model_dim, n_class, training_phase, dropout_p, pool)
            })
        )
    
    def forward(self, x):
        return self.vit(x)


if __name__ == "__main__":
    kwargs = {
        'p': 16,
        'model_dim': 768,
        'hidden_dim': 3072,
        'n_class': 1000,
        'n_heads': 12,
        'n_layers': 12,
        'n_patches': 196,
        #'batch_size': 10
    }
    vit = ViT(**kwargs)
    print(vit)

    # ip = torch.randn(4,196,768)
    # op = vit(ip)
    # print(op.shape)
    
    params = sum(p.numel() for p in vit.parameters())
    print(params)