# Transformer Encoder Layer makes NaN
# https://github.com/pytorch/pytorch/issues/24816
import sys
sys.path.append("..")
from common import *


class PositionEncode1D(nn.Module):
    def __init__(self, dim, max_length):
        super().__init__()
        assert (dim % 2 == 0)
        self.max_length = max_length

        d = torch.exp(torch.arange(0., dim, 2)* (-math.log(10000.0) / dim))
        position = torch.arange(0., max_length).unsqueeze(1)
        pos = torch.zeros(1, max_length, dim)
        pos[0, :, 0::2] = torch.sin(position * d)
        pos[0, :, 1::2] = torch.cos(position * d)
        self.register_buffer('pos', pos)

    def forward(self, x):
        batch_size, T, dim = x.shape
        x = x + self.pos[:,:T]
        return x

#https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
class PositionEncode2D(nn.Module):
    def __init__(self, dim, width, height):
        super().__init__()
        assert (dim % 4 == 0)
        self.width  = width
        self.height = height

        dim = dim//2
        d = torch.exp(torch.arange(0., dim, 2) * -(math.log(10000.0) / dim))
        position_w = torch.arange(0., width ).unsqueeze(1)
        position_h = torch.arange(0., height).unsqueeze(1)
        pos = torch.zeros(1, dim*2, height, width)

        pos[0,      0:dim:2, :, :] = torch.sin(position_w * d).transpose(0, 1).unsqueeze(1).repeat(1,1, height, 1)
        pos[0,      1:dim:2, :, :] = torch.cos(position_w * d).transpose(0, 1).unsqueeze(1).repeat(1,1, height, 1)
        pos[0,dim + 0:   :2, :, :] = torch.sin(position_h * d).transpose(0, 1).unsqueeze(2).repeat(1,1, 1, width)
        pos[0,dim + 1:   :2, :, :] = torch.cos(position_h * d).transpose(0, 1).unsqueeze(2).repeat(1,1, 1, width)
        self.register_buffer('pos', pos)

    def forward(self, x):
        batch_size,C,H,W = x.shape
        x = x + self.pos[:,:,:H,:W]
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

#layer normalization
class Norm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.bias  = nn.Parameter(torch.zeros(dim))
        self.eps   = eps
    def forward(self, x):
        #return x
        z = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps)
        x = self.alpha*z + self.bias
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_head, dropout=0.1):
        super().__init__()

        self.dim = dim
        self.d_k = dim // num_head
        self.num_head = num_head
        self.dropout = dropout

        self.q = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def attention(self, q, k, v, mask):
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # torch.Size([8, 4, 10, 10]) = batch_size, num_head, LqxLk

        if mask is not None:
            mask = mask.unsqueeze(1)
            score = score.masked_fill(mask == 0, -6e4)

        score = F.softmax(score, dim=-1)

        if self.dropout > 0:
            score = F.dropout(score, self.dropout, training=self.training)

        value = torch.matmul(score, v)
        return value


    def forward(self, q, k, v, mask=None):
        batch_size, T, dim = q.shape

        # perform linear operation and split into h heads
        k = self.k(k).reshape(batch_size, -1, self.num_head, self.d_k)
        q = self.q(q).reshape(batch_size, -1, self.num_head, self.d_k)
        v = self.v(v).reshape(batch_size, -1, self.num_head, self.d_k)

        # transpose to get dimensions batch_size * num_head * T * d_k
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        value = self.attention(q, k, v, mask)
        value = value.masked_fill(torch.isnan(value), 0)
        # concatenate heads and put through final linear layer
        value = value.transpose(1, 2).contiguous().reshape(batch_size, -1, self.dim)
        value = self.out(value)

        return value


#---
class TransformerEncodeLayer(nn.Module):
    def __init__(self, dim, ff_dim, num_head, dropout=0.1):
        super().__init__()
        self.norm1 = Norm(dim)
        self.norm2 = Norm(dim)

        self.attn = MultiHeadAttention(dim, num_head, dropout=0.1)
        self.ff   = FeedForward(dim, ff_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, x_mask):
        x1 = self.norm1(x)
        x1 = self.attn(x1, x1, x1, x_mask) #self-attention
        x   = x + self.dropout1(x1)

        x2 = self.norm2(x)
        x2 = self.ff(x2)
        x  = x + self.dropout2(x2)
        return x

class TransformerEncode(nn.Module):
    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__()
        self.num_layer = num_layer

        self.layer = nn.ModuleList([
            TransformerEncodeLayer(dim, ff_dim, num_head) for i in range(num_layer)
        ])
        self.norm = Norm(dim)

    def forward(self, x, x_mask):
        for i in range(self.num_layer):
            x = self.layer[i](x, x_mask)
        x = self.norm(x)
        return x

#---
class TransformerDecodeLayer(nn.Module):
    def __init__(self, dim, ff_dim, num_head, dropout=0.1):
        super().__init__()
        self.norm1 = Norm(dim)
        self.norm2 = Norm(dim)
        self.norm3 = Norm(dim)

        self.attn1 = MultiHeadAttention(dim, num_head, dropout=0.1)
        self.attn2 = MultiHeadAttention(dim, num_head, dropout=0.1)
        self.ff = FeedForward(dim, ff_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, mem, x_mask, mem_mask):
        x1 = self.norm1(x)
        x1 = self.attn1(x1, x1, x1, x_mask)  # self-attention
        x  = x + self.dropout1(x1)

        if mem is not None:
            x2 = self.norm2(x)
            x2 = self.attn2(x2, mem, mem, mem_mask)  # encoder input
            x  = x + self.dropout2(x2)

        x3 = self.norm3(x)
        x3 = self.ff(x3)
        x  = x + self.dropout3(x3)
        return x

    def forward_last(self, x_last, x_cache, mem, mem_mask):

        x_last_norm = self.norm1(x_last)
        x1 = torch.cat([x_cache, x_last_norm], 1)
        x_cache = x1.clone() # update

        x1 = self.attn1(x_last_norm, x1, x1)
        x_last  = x_last + x1

        if mem is not None:
            x2 = self.norm2(x_last)
            x2 = self.attn2(x2, mem, mem, mem_mask)
            x_last = x_last + x2


        x3 = self.norm3(x_last)
        x3 = self.ff(x3)
        x_last = x_last + x3

        return x_last, x_cache




# https://github.com/alexmt-scale/causal-transformer-decoder/blob/master/causal_transformer_decoder/model.py
class TransformerDecode(nn.Module):
    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__()
        self.num_layer = num_layer

        self.layer = nn.ModuleList([
            TransformerDecodeLayer(dim, ff_dim, num_head) for i in range(num_layer)
        ])
        self.norm = Norm(dim)

    def forward(self, x, mem, x_mask=None, mem_mask=None):

        for i in range(self.num_layer):
            x = self.layer[i](x, mem, x_mask, mem_mask)

        x = self.norm(x)
        return x

    def forward_last(self, x_last, x_cache, mem,  mem_mask=None):
        batch_size,t,dim = x_last.shape
        assert(t==1)

        for i in range(self.num_layer):
            x_last, x_cache[i] = self.layer[i].forward_last(x_last, x_cache[i], mem, mem_mask)

        x_last = self.norm(x_last)
        return x_last, x_cache




