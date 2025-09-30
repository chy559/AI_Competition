import torch

# 数据配置
DATA_CONFIG = {
    "train_path": "data/train.csv",
    "test_path": "data/test.csv",
    "label_col": "label",  # 标签列名
    "id_col": "id",        # 样本ID列（如有）
    "continuous_cols": ["age", "income"],  # 连续特征列名
    "categorical_cols": ["gender", "occupation", "city"],  # 离散特征列名
    "batch_size": 256,
    "test_size": 0.2
}

# 模型配置
MODEL_CONFIG = {
    "embedding_dim": 16,  # 离散特征嵌入维度
    "dnn_hidden_dims": [256, 128, 64],  # DNN隐藏层维度
    "dnn_dropout": 0.3,   # DNN dropout概率
    "fm_first_order": True  # 是否使用FM一阶项
}

# 训练配置
TRAIN_CONFIG = {
    "epochs": 50,
    "lr": 0.001,
    "weight_decay": 1e-5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": "deepfm_model.pth"
}