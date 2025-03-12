import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
class EncoderNoResidual(Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu, mlp_hidden_dims=[256, 128]):
        super(EncoderNoResidual, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act

        self.weight1 = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight2 = Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

        # Keep attention layer
        self.attention = AttentionLayer(out_features, out_features)
        
        # Remove residual layer
        # self.residual = ResidualLayer(out_features, out_features)  # Removed
        
        # Keep MLP layer
        self.mlp = MLP(out_features, mlp_hidden_dims, out_features, dropout, act)
        self.disc = DiscriminatorWithMLP(out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        # First part: encoder computes hidden representation
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = self.attention(z, adj)
        
        # Skip residual connection
        # z = self.residual(z)  # Removed
        
        z = self.mlp(z)
        hiden_emb = z

        h = torch.mm(z, self.weight2)
        h = torch.spmm(adj, h)
        emb = self.act(z)

        # Process alternative input
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = self.attention(z_a, adj)
        
        # Skip residual connection
        # z_a = self.residual(z_a)  # Removed
        
        z_a = self.mlp(z_a)
        emb_a = self.act(z_a)

        # Calculate global representation
        g = self.read(emb, self.graph_neigh)
        g = self.sigm(g)

        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)

        # Discriminator operations
        ret = self.disc(g, emb, emb_a)
        ret_a = self.disc(g_a, emb_a, emb)

        return hiden_emb, h, ret, ret_a

class EncoderNoMLP(Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(EncoderNoMLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act

        self.weight1 = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight2 = Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

        # Keep attention and residual layers
        self.attention = AttentionLayer(out_features, out_features)
        self.residual = ResidualLayer(out_features, out_features)
        
        # Remove MLP layer
        # self.mlp = MLP(out_features, mlp_hidden_dims, out_features, dropout, act)  # Removed
        
        # Use standard discriminator instead of MLP-enhanced one
        self.disc = Discriminator(out_features)  # Changed from DiscriminatorWithMLP
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        # First part: encoder computes hidden representation
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = self.attention(z, adj)
        z = self.residual(z)
        
        # Skip MLP layer
        # z = self.mlp(z)  # Removed
        
        hiden_emb = z

        h = torch.mm(z, self.weight2)
        h = torch.spmm(adj, h)
        emb = self.act(z)

        # Process alternative input
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = self.attention(z_a, adj)
        z_a = self.residual(z_a)
        
        # Skip MLP layer
        # z_a = self.mlp(z_a)  # Removed
        
        emb_a = self.act(z_a)

        # Calculate global representation
        g = self.read(emb, self.graph_neigh)
        g = self.sigm(g)

        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)

        # Discriminator operations
        ret = self.disc(g, emb, emb_a)
        ret_a = self.disc(g_a, emb_a, emb)

        return hiden_emb, h, ret, ret_a


class EncoderNoAttention(Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu, mlp_hidden_dims=[256, 128]):
        super(EncoderNoAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act

        self.weight1 = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight2 = Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

        # Remove attention layer and replace with simple matrix multiplication
        # self.attention = AttentionLayer(out_features, out_features)  # Removed
        
        # Keep residual and MLP layers
        self.residual = ResidualLayer(out_features, out_features)
        self.mlp = MLP(out_features, mlp_hidden_dims, out_features, dropout, act)
        self.disc = DiscriminatorWithMLP(out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        # First part: encoder computes hidden representation
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        
        # Replace attention with simple matrix multiplication
        z = torch.spmm(adj, z)  # Simple graph convolution instead of attention
        
        z = self.residual(z)
        z = self.mlp(z)
        hiden_emb = z

        h = torch.mm(z, self.weight2)
        h = torch.spmm(adj, h)
        emb = self.act(z)

        # Process alternative input
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        
        # Replace attention with simple matrix multiplication
        z_a = torch.spmm(adj, z_a)
        
        z_a = self.residual(z_a)
        z_a = self.mlp(z_a)
        emb_a = self.act(z_a)

        # Calculate global representation
        g = self.read(emb, self.graph_neigh)
        g = self.sigm(g)

        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)

        # Discriminator operations
        ret = self.disc(g, emb, emb_a)
        ret_a = self.disc(g_a, emb_a, emb)

        return hiden_emb, h, ret, ret_a


class AttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(AttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a_src = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.a_dst = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W)
        torch.nn.init.xavier_uniform_(self.a_src)
        torch.nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # Linear transformation
        attn_src = torch.mm(Wh, self.a_src)  # Attention coefficient from src
        attn_dst = torch.mm(Wh, self.a_dst)  # Attention coefficient from dst
        
        e = attn_src + attn_dst.T  # Pairwise attention (broadcasting for all pairs)
        e = self.leaky_relu(e)

        # Mask attention with adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)  # Large negative value to mask zeros
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        h_prime = torch.mm(attention, Wh)
        return h_prime





class ResidualLayer(nn.Module):
    def __init__(self, in_features, out_features, act=F.relu):
        super(ResidualLayer, self).__init__()
        self.act = act

        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, in_features)

    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = self.act(out)
        out = self.linear2(out)

        # Ensure dimensions match for residual
        if out.shape != identity.shape:
            identity = nn.Linear(identity.shape[1], out.shape[1]).to(identity.device)(identity)

        out += identity
        return self.act(out)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0, act=F.relu):
        super(MLP, self).__init__()
        self.act = act
        self.layers = nn.ModuleList()

        in_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(in_dim, h_dim))
            self.layers.append(nn.Dropout(dropout))  # Add dropout
            in_dim = h_dim

        self.layers.append(nn.Linear(in_dim, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x)  # Output layer without activation
        return x
    
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
    
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum 
          
        return F.normalize(global_emb, p=2, dim=1) 
    

class DiscriminatorWithMLP(nn.Module):
    def __init__(self, n_h):
        super(DiscriminatorWithMLP, self).__init__()
        self.mlp = MLP(n_h, [128], n_h)  # 在判别器之前加一层 MLP
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c = self.mlp(c)  # 用 MLP 变换 c
        h_pl = self.mlp(h_pl)  # 用 MLP 变换正样本
        h_mi = self.mlp(h_mi)  # 用 MLP 变换负样本
        
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

class EncoderAttentionResidualMLP(Module):
    def __init__(
        self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu, mlp_hidden_dims=[256, 128]
    ):
        super(EncoderAttentionResidualMLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act

        self.weight1 = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight2 = Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

        self.attention = AttentionLayer(out_features, out_features)
        self.residual = ResidualLayer(out_features, out_features)

        # 添加 MLP 层
        self.mlp = MLP(out_features, mlp_hidden_dims, out_features, dropout, act)

        self.disc = DiscriminatorWithMLP(out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        # 第一部分：编码器计算隐藏表示
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = self.attention(z, adj)
        z = self.residual(z)

        # 将隐藏表示传递给 MLP
        z = self.mlp(z)

        hiden_emb = z

        h = torch.mm(z, self.weight2)
        h = torch.spmm(adj, h)

        emb = self.act(z)

        # 对备用输入进行相同处理
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = self.attention(z_a, adj)
        z_a = self.residual(z_a)

        # 通过 MLP 对备用输入的隐藏表示进行变换
        z_a = self.mlp(z_a)

        emb_a = self.act(z_a)

        # 计算全局表示
        g = self.read(emb, self.graph_neigh)
        g = self.sigm(g)

        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)

        # 判别器操作
        ret = self.disc(g, emb, emb_a)
        ret_a = self.disc(g_a, emb_a, emb)

        return hiden_emb, h, ret, ret_a

class EncoderSparseAttentionResidual(Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(EncoderSparseAttentionResidual, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act

        self.weight1 = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight2 = Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

        self.attention = AttentionLayer(out_features, out_features)
        self.residual = ResidualLayer(out_features, out_features)

        self.disc = Discriminator(out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = self.attention(z, adj)
        z = self.residual(z)

        hiden_emb = z

        h = torch.mm(z, self.weight2)
        h = torch.spmm(adj, h)

        emb = self.act(z)

        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = self.attention(z_a, adj)
        z_a = self.residual(z_a)

        emb_a = self.act(z_a)

        g = self.read(emb, self.graph_neigh)
        g = self.sigm(g)

        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)

        ret = self.disc(g, emb, emb_a)
        ret_a = self.disc(g_a, emb_a, emb)

        return hiden_emb, h, ret, ret_a
class EncoderSparseAttentionResidual(Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(EncoderSparseAttentionResidual, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act

        # 初始化权重参数
        self.weight1 = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight2 = Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

        # 图注意力层和残差层
        self.attention = AttentionLayer(out_features, out_features)
        self.residual = ResidualLayer(out_features, out_features)

        # 判别器和读取器
        self.disc = Discriminator(out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        # 处理输入特征 feat
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)  # 特征线性变换
        z = self.attention(z, adj)  # 图注意力处理
        z = self.residual(z)  # 残差连接
        hiden_emb = z

        # 处理稀疏矩阵计算
        h = torch.mm(z, self.weight2)  # 第二次线性变换
        h = torch.spmm(adj, h)  # 稀疏矩阵相乘

        emb = self.act(z)  # 激活嵌入特征

        # 处理对抗特征 feat_a
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)  # 特征线性变换
        z_a = self.attention(z_a, adj)  # 图注意力处理
        z_a = self.residual(z_a)  # 残差连接

        emb_a = self.act(z_a)  # 激活对抗嵌入

        # 读取全局图特征
        g = self.read(emb, self.graph_neigh)
        g = self.sigm(g)

        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)

        # 判别器输出
        ret = self.disc(g, emb, emb_a)
        ret_a = self.disc(g_a, emb_a, emb)

        return hiden_emb, h, ret, ret_a
 
class Encoder(Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()
        
        self.disc = Discriminator(self.out_features)

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.mm(adj, z)
        
        hiden_emb = z
        
        h = torch.mm(z, self.weight2)
        h = torch.mm(adj, h)
        
        emb = self.act(z)
        
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.mm(adj, z_a)
        emb_a = self.act(z_a)
        
        g = self.read(emb, self.graph_neigh) 
        g = self.sigm(g)  

        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)  

        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb) 
        
        return hiden_emb, h, ret, ret_a
#大规模    
class Encoder_sparse(Module):
    """
    Sparse version of Encoder
    """
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(Encoder_sparse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()
        
        self.disc = Discriminator(self.out_features)

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.spmm(adj, z)
        
        hiden_emb = z
        
        h = torch.mm(z, self.weight2)
        h = torch.spmm(adj, h)
        
        emb = self.act(z)
        
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.spmm(adj, z_a)
        emb_a = self.act(z_a)
         
        g = self.read(emb, self.graph_neigh)
        g = self.sigm(g)
        
        g_a = self.read(emb_a, self.graph_neigh)
        g_a =self.sigm(g_a)       
       
        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb)
        
        return hiden_emb, h, ret, ret_a     

class Encoder_sc(torch.nn.Module):
    def __init__(self, dim_input, dim_output, dropout=0.0, act=F.relu):
        super(Encoder_sc, self).__init__()
        self.dim_input = dim_input
        self.dim1 = 256
        self.dim2 = 64
        self.dim3 = 32
        self.act = act
        self.dropout = dropout
        
        
        self.weight1_en = Parameter(torch.FloatTensor(self.dim_input, self.dim1))
        self.weight2_en = Parameter(torch.FloatTensor(self.dim1, self.dim2))
        self.weight3_en = Parameter(torch.FloatTensor(self.dim2, self.dim3))
        
        self.weight1_de = Parameter(torch.FloatTensor(self.dim3, self.dim2))
        self.weight2_de = Parameter(torch.FloatTensor(self.dim2, self.dim1))
        self.weight3_de = Parameter(torch.FloatTensor(self.dim1, self.dim_input))
      
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1_en)
        torch.nn.init.xavier_uniform_(self.weight1_de)
        
        torch.nn.init.xavier_uniform_(self.weight2_en)
        torch.nn.init.xavier_uniform_(self.weight2_de)
        
        torch.nn.init.xavier_uniform_(self.weight3_en)
        torch.nn.init.xavier_uniform_(self.weight3_de)
        
    def forward(self, x):
        x = F.dropout(x, self.dropout, self.training)
        
        
        x = torch.mm(x, self.weight1_en)
        x = torch.mm(x, self.weight2_en)
        x = torch.mm(x, self.weight3_en)
        
        x = torch.mm(x, self.weight1_de)
        x = torch.mm(x, self.weight2_de)
        x = torch.mm(x, self.weight3_de)
        
        return x
    
class Encoder_map(torch.nn.Module):
    def __init__(self, n_cell, n_spot):
        super(Encoder_map, self).__init__()
        self.n_cell = n_cell
        self.n_spot = n_spot
          
        self.M = Parameter(torch.FloatTensor(self.n_cell, self.n_spot))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.M)
        
    def forward(self):
        x = self.M
        
        return x 