import torch.nn
import torch


class FullyConnectedOutput(torch.nn.Module):
    def __init__(self, embed_dim, input_dim):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 32),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, embed_dim),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(p=0.1)
        )

        self.norm = torch.nn.LayerNorm(normalized_shape=embed_dim, elementwise_affine=True)

    def forward(self, x):
        # 规范化
        x = self.norm(x)

        # 线性全连接
        # [b, 50, 32] -> [b, 50, 32]
        out = self.fc(x)

        return out


# 注意力计算函数
def attention(Q, K, V):
    # b句话，每句话50个词，每个词32维，4个头，每个头8维
    # Q,K,V = [b, 4, 50, 8]
    l = Q.shape[2]
    num_head = Q.shape[1]

    # Q, K相乘，求每个词对其他所有词的注意力
    # [b, 4, 50, 8] * [b, 4, 8, 50] -> [b, 4, 50, 50]
    score = torch.matmul(Q, K.permute(0, 1, 3, 2))

    # 除以每个头尾数的平凡根，做数值缩放
    score /= 8 ** 0.5

    score = torch.softmax(score, dim=-1)

    # score * V, 得到最后注意力的结果
    # [b, 4, 50, 50] * [b, 4, 50, 8] -> [b, 4, 50, 8]
    score = torch.matmul(score, V)

    # 每个头的计算合一
    # [b, 4, 50, 8] -> [b, 50, 32]
    score = score.permute(0, 2, 1, 3).reshape(-1, l, num_head * Q.shape[3])

    return score


class MultiHead(torch.nn.Module):
    def __init__(self, input_dim, num_head, embed_dim):
        super().__init__()
        self.fc_Q = torch.nn.Linear(input_dim, 32)
        self.fc_K = torch.nn.Linear(input_dim, 32)
        self.fc_V = torch.nn.Linear(input_dim, 32)

        self.num_head = num_head

        self.out_fc = torch.nn.Linear(32, embed_dim)

        self.norm = torch.nn.LayerNorm(normalized_shape=input_dim, elementwise_affine=True)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, Q, K, V):
        # b句话，50个词，每个词32维向量
        # Q, K, V = [b, 50, 32]
        b = Q.shape[0]
        len = Q.shape[1]

        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)

        # 线性运算，维度不变
        # [b, 50, 32] -> [b, 50, 32]
        K = self.fc_K(K)
        V = self.fc_V(V)
        Q = self.fc_Q(Q)

        # 拆分成多个头
        # [b, 50, 32] -> [b, 4, 50, 8]
        Q = Q.reshape(b, len, self.num_head, -1).permute(0, 2, 1, 3)
        K = K.reshape(b, len, self.num_head, -1).permute(0, 2, 1, 3)
        V = V.reshape(b, len, self.num_head, -1).permute(0, 2, 1, 3)

        # 计算注意力
        # [b, 4, 50, 8] -> [b, 50, 32]
        score = attention(Q, K, V)

        # 计算输出，维度不变
        # [b, 50, 32] -> [b, 50, 32]
        score = self.dropout(self.out_fc(score))

        return score


class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, num_head, embed_dim):
        super(EncoderLayer, self).__init__()
        self.mh = MultiHead(input_dim, num_head, embed_dim)
        self.fc = FullyConnectedOutput(embed_dim, input_dim)

    def forward(self, x):
        score = self.mh(x, x, x)
        out = self.fc(score)

        return out


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, num_head, embed_dim):
        super(Encoder, self).__init__()
        self.layer = EncoderLayer(input_dim, num_head, embed_dim)

    def forward(self, x):
        x = self.layer(x)

        return x
