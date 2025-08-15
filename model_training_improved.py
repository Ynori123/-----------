import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用的GPU


# 设置matplotlib使用英文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用支持英文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 改进的DataFrameDataset类
class DataFrameDataset(Dataset):
    def __init__(self, dataframe, input_columns, target_column):
        self.dataframe = dataframe.copy()

        # 检查并处理NaN值
        if self.dataframe[input_columns].isnull().any().any():
            print("Warning: Input data contains NaN values, replacing with 0")
            self.dataframe[input_columns] = self.dataframe[input_columns].fillna(0)

        self.inputs = self.dataframe[input_columns].values
        self.targets = self.dataframe[target_column].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y

# 改进的MLP模型 - 添加批归一化和Dropout
class ImprovedMLP(L.LightningModule):
    def __init__(self, input_size, hidden_size=128, output_size=2, dropout_rate=0.3, learning_rate=0.0005):
        super(ImprovedMLP, self).__init__()
        self.save_hyperparameters()

        # 增加网络深度
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # 添加批归一化
        self.dropout1 = nn.Dropout(dropout_rate)  # 添加Dropout

        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.layer4 = nn.Linear(hidden_size // 2, output_size)

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=output_size)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=output_size)

        # 添加梯度裁剪参数
        self.gradient_clip_val = 1.0

    def forward(self, x):
        # 前向传播，添加批归一化和Dropout
        x = F.relu(self.bn1(self.layer1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.layer2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.layer3(x)))
        x = self.dropout3(x)

        x = self.layer4(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_acc(logits, y)

        # 添加L2正则化
        l2_reg = None
        for param in self.parameters():
            if param.requires_grad:
                if l2_reg is None:
                    l2_reg = torch.norm(param, 2)
                else:
                    l2_reg = l2_reg + torch.norm(param, 2)

        # 添加权重衰减
        weight_decay = 1e-5
        loss = loss + weight_decay * l2_reg

        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_acc(logits, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # 使用AdamW优化器，内置权重衰减
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-5)

        # 使用余弦退火学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

        return [optimizer], [scheduler]

# 改进的数据预处理函数
def preprocess_data(df, input_columns, target_column):
    """
    改进的数据预处理函数，包括更严格的NaN处理和数据分割
    """
    # 创建数据的副本以避免修改原始数据
    df_processed = df.copy()

    # 检查目标列的类别分布
    print("目标列类别分布:")
    print(df_processed[target_column].value_counts(normalize=True))

    # 检查并处理NaN值
    print("检查数据中的NaN值...")
    if df_processed[input_columns].isnull().any().any():
        print("警告: 输入数据中包含NaN值，将替换为0")
        df_processed[input_columns] = df_processed[input_columns].fillna(0)

    # 检查目标列中是否有NaN值
    if df_processed[target_column].isnull().any():
        print(f"警告: 目标列中包含NaN值，将删除这些行")
        df_processed = df_processed.dropna(subset=[target_column])

    # 标准化数据
    print("标准化数据...")
    scaler = StandardScaler()
    trans_columns = [c + '_trans' for c in input_columns]
    df_processed[trans_columns] = scaler.fit_transform(df_processed[input_columns])

    # 检查并处理类别不平衡问题
    print("检查类别不平衡...")
    if len(df_processed[target_column].unique()) > 2:
        print("目标列包含多个类别，将使用分层抽样")
        # 对于多类别，使用分层抽样需要特殊处理
        df_processed = df_processed.sample(frac=1.0, random_state=42)
    else:
        print("目标列包含两个类别，将使用分层抽样")
        # 对于二分类，使用train_test_split进行分层抽样
        # 对于二分类，直接保持数据顺序，不进行采样
        # 因为后续的train_test_split会进行分层抽样
        # df_processed = train_df
        pass
    print("数据预处理完成")

    return df_processed

# 改进的训练函数
def train_model(df, input_columns, target_column, model_path, max_epochs=50, batch_size=256):
    """
    改进的模型训练函数，包含更好的数据处理和训练策略
    """
    # 添加GPU诊断
    print("PyTorch版本:", torch.__version__)
    print("CUDA是否可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA版本:", torch.version.cuda)
        print("GPU数量:", torch.cuda.device_count())
        print("当前GPU:", torch.cuda.current_device())
        print("GPU名称:", torch.cuda.get_device_name(0))

    # 添加设备检测
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"检测到 {num_gpus} 个GPU设备")
        accelerator = "gpu"
        devices = min(num_gpus, 2)  # 最多使用2个GPU
        strategy = "ddp_spawn"
    else:
        print("未检测到GPU设备 使用CPU")
        accelerator = "cpu"
        devices = 1
        strategy = "auto"

    # 创建训练器时指定设备
    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=L.pytorch.loggers.TensorBoardLogger('tb_logs/improved', name='Z'),
        gradient_clip_val=1.0,
        callbacks=[L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")],
        accelerator=accelerator,
        devices=devices,
        strategy=strategy
    )

        # 创建模型时移动到指定设备
    model = ImprovedMLP(input_size=len(input_columns), hidden_size=128, output_size=2)
    if accelerator == "gpu":
        model = model.to(device)
        if num_gpus > 1:
            model = nn.DataParallel(model)




    # 数据预处理
    df_processed = preprocess_data(df, input_columns, target_column)
    trans_columns = [c + '_trans' for c in input_columns]
    # 检查目标列的类别分布
    print("目标列类别分布:")
    print(df_processed[target_column].value_counts(normalize=True))

    # 检查数据完整性
    print(f"处理前数据形状: {df_processed.shape}")
    print(f"包含NaN值的列: {df_processed.isnull().sum()[df_processed.isnull().sum() > 0].index.tolist()}")

    # 分割数据集，使用分层抽样
    train_df, test_df = train_test_split(
        df_processed, 
        test_size=0.2, 
        random_state=42,
        stratify=df_processed[target_column]
    )
    val_df, test_df = train_test_split(
        test_df, 
        test_size=0.5, 
        random_state=42,
        stratify=test_df[target_column]
    )

    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"测试集大小: {len(test_df)}")

    # 创建数据集
    train_dataset = DataFrameDataset(train_df, trans_columns, target_column)
    val_dataset = DataFrameDataset(val_df, trans_columns, target_column)
    test_dataset = DataFrameDataset(test_df, trans_columns, target_column)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, persistent_workers=True)

    # 创建模型
    model = ImprovedMLP(input_size=len(input_columns), hidden_size=128, output_size=2)

    # 创建训练器，启用梯度裁剪
    trainer = L.Trainer(
        max_epochs=max_epochs, 
        logger=L.pytorch.loggers.TensorBoardLogger('tb_logs/improved', name='Z'),
        gradient_clip_val=1.0,  # 添加梯度裁剪
        callbacks=[L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")]
    )

    # 训练模型
    print("开始训练模型...")
    trainer.fit(model, train_loader, val_loader)

    # 保存模型
    print(f"保存模型到: {model_path}")
    trainer.save_checkpoint(model_path)

    # 在测试集上评估模型
    print("在测试集上评估模型...")
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            pos_probs = probs[:, 1].cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(pos_probs)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 计算并打印分类报告
    preds = (all_probs > 0.5).astype(int)
    class_report = classification_report(all_labels, preds, target_names=["Background", "Signal"], zero_division=0)
    print("\n分类报告:")
    print(class_report)

    # 计算AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    print(f"测试集AUC: {roc_auc:.4f}")

    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.yscale('log')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)\nImproved Model Performance')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curve_improved.png', dpi=300, bbox_inches='tight')
    plt.show()

    return model, roc_auc




# 使用示例
if __name__ == "__main__":
    # 请根据实际情况修改这些参数
    model_path = "./tb_logs/Z/version_improved/best_model.ckpt"

    # 假设这些变量已经在您的notebook中定义
    # 如果没有，请在这里定义
    # input_columns = [...]
    # target_column = ...
    # df = ...

    # 训练模型
    model, auc = train_model(
        df=df,
        input_columns=input_columns,
        target_column=target_column,
        model_path=model_path,
        max_epochs=50,
        batch_size=256
    )

    print(f"训练完成，测试集AUC: {auc:.4f}")
