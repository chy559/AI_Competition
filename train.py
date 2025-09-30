import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.deepfm import DeepFM
from utils.data_processor import get_dataloaders
from utils.metrics import calculate_metrics
from config import DATA_CONFIG, TRAIN_CONFIG, MODEL_CONFIG

def train():
    # 加载数据
    train_loader, val_loader, processor = get_dataloaders()
    num_cont_features = len(DATA_CONFIG["continuous_cols"])
    cat_dims = processor.get_cat_dims()
    
    # 初始化模型
    model = DeepFM(num_cont_features, cat_dims).to(TRAIN_CONFIG["device"])
    criterion = nn.BCELoss()  # 二分类交叉熵
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"]
    )
    
    # 训练循环
    best_auc = 0.0
    for epoch in range(TRAIN_CONFIG["epochs"]):
        model.train()
        train_loss = 0.0
        for cont, cat, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['epochs']}"):
            cont, cat, label = cont.to(TRAIN_CONFIG["device"]), cat.to(TRAIN_CONFIG["device"]), label.to(TRAIN_CONFIG["device"])
            
            optimizer.zero_grad()
            pred = model(cont, cat).squeeze()
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * cont.size(0)
        
        # 验证
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for cont, cat, label in val_loader:
                cont, cat, label = cont.to(TRAIN_CONFIG["device"]), cat.to(TRAIN_CONFIG["device"]), label.to(TRAIN_CONFIG["device"])
                pred = model(cont, cat).squeeze()
                loss = criterion(pred, label)
                val_loss += loss.item() * cont.size(0)
                
                all_labels.append(label)
                all_preds.append(pred)
        
        # 计算指标
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        metrics = calculate_metrics(all_labels, all_preds)
        
        # 打印结果
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Metrics: AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
        
        # 保存最佳模型
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(model.state_dict(), TRAIN_CONFIG["save_path"])
            print(f"Saved best model (AUC: {best_auc:.4f})")

if __name__ == "__main__":
    train()