import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MODEL_CONFIG

class DeepFM(nn.Module):
    def __init__(self, num_cont_features, cat_dims):
        super(DeepFM, self).__init__()
        self.embedding_dim = MODEL_CONFIG["embedding_dim"]
        self.fm_first_order = MODEL_CONFIG["fm_first_order"]
        
        # 1. FM部分
        # 1.1 一阶特征（连续特征 + 离散特征）
        if self.fm_first_order:
            self.cont_first_order = nn.Linear(num_cont_features, 1)  # 连续特征一阶
            self.cat_first_order = nn.ModuleList([
                nn.Embedding(cat_dim, 1) for cat_dim in cat_dims  # 离散特征一阶（嵌入维度1）
            ])
        
        # 1.2 二阶特征（仅离散特征，通过嵌入计算交互）
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, self.embedding_dim) for cat_dim in cat_dims  # 离散特征嵌入
        ])
        
        # 2. DNN部分
        # 输入维度：连续特征数 + 离散特征数 * 嵌入维度
        dnn_input_dim = num_cont_features + len(cat_dims) * self.embedding_dim
        # 构建DNN层
        self.dnn = nn.Sequential()
        for dim in MODEL_CONFIG["dnn_hidden_dims"]:
            self.dnn.add_module(f"linear_{dim}", nn.Linear(dnn_input_dim, dim))
            self.dnn.add_module(f"relu_{dim}", nn.ReLU())
            self.dnn.add_module(f"dropout_{dim}", nn.Dropout(MODEL_CONFIG["dnn_dropout"]))
            dnn_input_dim = dim
        
        # 3. 输出层（结合FM和DNN结果）
        final_input_dim = 1  # FM二阶输出1维
        if self.fm_first_order:
            final_input_dim += 1  # 加上FM一阶输出1维
        final_input_dim += MODEL_CONFIG["dnn_hidden_dims"][-1]  # 加上DNN输出
        self.final_linear = nn.Linear(final_input_dim, 1)
        self.sigmoid = nn.Sigmoid()  # 二分类用

    def forward(self, cont_features, cat_features):
        # FM一阶部分
        fm_first = 0.0
        if self.fm_first_order:
            # 连续特征一阶
            fm_first += self.cont_first_order(cont_features)
            # 离散特征一阶
            for i in range(cat_features.shape[1]):
                fm_first += self.cat_first_order[i](cat_features[:, i])
        
        # FM二阶部分（仅离散特征）
        cat_embeds = []
        for i in range(cat_features.shape[1]):
            embed = self.cat_embeddings[i](cat_features[:, i])  # (batch_size, embedding_dim)
            cat_embeds.append(embed)
        cat_embeds = torch.stack(cat_embeds, dim=1)  # (batch_size, num_cat, embedding_dim)
        
        # 计算二阶交互：sum(square) - square(sum)
        sum_square = torch.sum(cat_embeds, dim=1) **2  # (batch_size, embedding_dim)
        square_sum = torch.sum(cat_embeds** 2, dim=1)  # (batch_size, embedding_dim)
        fm_second = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)  # (batch_size, 1)
        
        # DNN部分
        # 拼接连续特征和离散特征嵌入
        dnn_input = torch.cat([cont_features] + cat_embeds.unbind(dim=1), dim=1)  # (batch_size, dnn_input_dim)
        dnn_out = self.dnn(dnn_input)  # (batch_size, last_dnn_dim)
        
        # 结合所有部分
        if self.fm_first_order:
            total_input = torch.cat([fm_first, fm_second, dnn_out], dim=1)
        else:
            total_input = torch.cat([fm_second, dnn_out], dim=1)
        output = self.final_linear(total_input)
        return self.sigmoid(output)  # 二分类概率