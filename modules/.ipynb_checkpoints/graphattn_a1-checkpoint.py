import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Dense(nn.Module):

    def __init__(self, in_features, out_features, activation=F.relu):
        super(Dense, self).__init__()
        self.fc = torch.nn.Linear(in_features, out_features)
        self.bn = torch.nn.BatchNorm1d(num_features=out_features)
        self.activation = activation

    def forward(self,x):
        x = self.bn(self.fc(x))
        return self.activation(x)
    
class ResBlock(nn.Module):

    def __init__(self, in_features, activation=F.relu):
        super(ResBlock, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, in_features)
        self.bn1 = torch.nn.BatchNorm1d(num_features=in_features)
        self.fc2 = torch.nn.Linear(in_features, in_features)
        self.bn2 = torch.nn.BatchNorm1d(num_features=in_features)
        self.activation = activation

    def forward(self,x):
        x1 = self.activation(self.bn1(self.fc1(x)))
        x2 = self.bn2(self.fc2(x1))
        return x+x2
    
    
class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None, mask_type="0-1"): #mask_type="soft", "0-1"
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        #print("scores.shape=", scores.shape, ", mask.shape=", mask.shape)
        if mask is not None:
            scores = scores.masked_fill_(mask == 0, -1e9)
            if mask_type == "soft":
                scores *= mask
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)

class MultiHeadAttention(nn.Module):
    """
    >>> import torch
    >>> from modules.graphattn_a1 import MultiHeadAttention
    >>> kp_refiner = MultiHeadAttention(2,2,1)

    >>> #try
    >>> bs = 5
    >>> x = torch.rand((bs,10,2))
    >>> kp_refiner.assign_mask_weight(adjmatrix_directed)
    >>> y, a, _ = kp_refiner(x, mask_type="soft")
    >>> print( y.shape, a.shape )
    >>> print( kp_refiner.mask_weight )
    torch.Size([5, 10, 2]) torch.Size([10, 10])
    Parameter containing:
    tensor([[1., 0., 0., 0., 1., 0., 0., 1., 1., 1.],
            [0., 1., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 1., 1., 0., 1., 0., 0., 1., 0.],
            [0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 1., 0.],
            [1., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [1., 1., 1., 0., 0., 0., 1., 0., 1., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], requires_grad=True)
    """

    def __init__(self,
                 in_features=2,
                 out_features=2, # added
                 head_num=1,
                 bias=True,
                 activation=F.relu,
                 alpha = 0.3,
                ):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, out_features, bias)
        
        # mask weight
        self.mask_weight = nn.Parameter(torch.rand(10,10).normal_(0.0, 0.02))
        self.alpha = alpha #nn.Parameter(torch.rand(1).normal_(0.1, 0.02))
        self.mask_trans_fn = lambda x: F.relu(x)
            
        
    def forward(self, x, mask_type="soft"):#mask_type="soft", "0-1"
        
        mask = None
        op_adj_w = self.mask_trans_fn(self.mask_weight)
        if mask_type=="soft":
            mask = op_adj_w #> 0.5
        
        q, k, v = x, x, x
        batch_size = q.size()[0]
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(batch_size*self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, mask, mask_type=mask_type)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        
        alpha = self.alpha #F.relu(self.alpha)+0.1
        
        return (1-alpha)*x+alpha*y, (op_adj_w>0.5).to(int), op_adj_w
    
    def assign_mask_weight(self, mask=None):
        if not mask is None:
            self.mask_weight.data = mask    
            
    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )