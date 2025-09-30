import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from config import DATA_CONFIG

class FeatureProcessor:
    def __init__(self):
        self.continuous_cols = DATA_CONFIG["continuous_cols"]
        self.categorical_cols = DATA_CONFIG["categorical_cols"]
        self.cont_scaler = MinMaxScaler()  # 连续特征归一化
        self.cat_encoders = {col: LabelEncoder() for col in self.categorical_cols}  # 离散特征编码
        self.cat_dims = {}  # 存储每个离散特征的类别数

    def fit(self, df):
        # 拟合连续特征
        self.cont_scaler.fit(df[self.continuous_cols])
        # 拟合离散特征
        for col in self.categorical_cols:
            # 填充缺失值（假设用"unknown"）
            df[col] = df[col].fillna("unknown").astype(str)
            self.cat_encoders[col].fit(df[col])
            self.cat_dims[col] = len(self.cat_encoders[col].classes_)

    def transform(self, df):
        # 处理连续特征
        cont_features = self.cont_scaler.transform(df[self.continuous_cols])
        # 处理离散特征
        cat_features = []
        for col in self.categorical_cols:
            df[col] = df[col].fillna("unknown").astype(str)
            # 处理未见过的类别
            mask = ~df[col].isin(self.cat_encoders[col].classes_)
            df.loc[mask, col] = "unknown"
            cat_encoded = self.cat_encoders[col].transform(df[col])
            cat_features.append(cat_encoded)
        cat_features = np.stack(cat_features, axis=1)
        return cont_features, cat_features

    def get_cat_dims(self):
        return list(self.cat_dims.values())

class CustomDataset(Dataset):
    def __init__(self, cont_features, cat_features, labels=None):
        self.cont_features = torch.FloatTensor(cont_features)
        self.cat_features = torch.LongTensor(cat_features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.cont_features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.cont_features[idx], self.cat_features[idx], self.labels[idx]
        return self.cont_features[idx], self.cat_features[idx]

def get_dataloaders():
    # 加载数据
    df = pd.read_csv(DATA_CONFIG["train_path"])
    X = df.drop(columns=[DATA_CONFIG["label_col"], DATA_CONFIG["id_col"]], errors="ignore")
    y = df[DATA_CONFIG["label_col"]].values

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=DATA_CONFIG["test_size"], random_state=42)

    # 特征预处理
    processor = FeatureProcessor()
    processor.fit(X_train)
    cont_train, cat_train = processor.transform(X_train)
    cont_val, cat_val = processor.transform(X_val)

    # 创建数据集和数据加载器
    train_dataset = CustomDataset(cont_train, cat_train, y_train)
    val_dataset = CustomDataset(cont_val, cat_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=DATA_CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=DATA_CONFIG["batch_size"], shuffle=False)

    return train_loader, val_loader, processor