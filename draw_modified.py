
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import model_training_improved as mti
import torch.nn.functional as F
import pytorch_lightning as L
from torch import nn
import torchmetrics
import awkward as ak
import os
import model_training_improved as mti
import model_training_cosine_annealing as mtca

# 定义基础MLP模型类
class SimpleMLP(L.LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)   # 第一层全连接层
        self.layer2 = nn.Linear(hidden_size, hidden_size)  # 第二层全连接层
        self.layer3 = nn.Linear(hidden_size, output_size)  # 输出层

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=output_size)  # 训练准确率指标
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=output_size)    # 验证准确率指标

    def forward(self, x):
        # 前向传播：输入 -> relu -> 隐藏层 -> relu -> 输出
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def training_step(self, batch, batch_idx):
        # 定义每个训练批次的操作
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_acc(logits, y)
        # 记录训练损失和准确率
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # 定义每个验证批次的操作
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_acc(logits, y)
        # 记录验证损失和准确率
        self.log('val_loss', loss)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # 设置优化器和学习率调度器
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

# 定义输入的列和目标列，每个模型包含名称、路径和线条样式
input_columns = ['jet_pt1', 'jet_eta1', 'jet_phi1', 'jet_mass1', 'jet_pt2', 
                'jet_eta2', 'jet_phi2', 'jet_mass2', 'jet_deta12', 'jet_dphi12',
                'jet_inv_mass', 'jet_dR12', 'jet_aT']
target_column = 'label_is_sig'

# 修改如果需要添加更多模型，新增即可，使用相对路径
# 需要包含模型训练完毕以后的checkpoint文件路径

# 定义模型列表，每个模型包含名称、路径和线条样式
models_info = [
    {
        "name": "4L simple cos_anl",
        "path": "./tb_logs/improved_model/16-56.ckpt",
        "color": "darkorange",
        "linestyle": "-",
        "model_class": mti.ImprovedMLP,
        "model_args": {"hidden_size": 128, "output_size": 2}
    },
    {
        "name": "3L",
        "path": "./tb_logs/simple_mlp_zprime/version_5/checkpoints/epoch=49-step=11750.ckpt",
        "color": "blue",
        "linestyle": "--",
        "model_class": SimpleMLP,
        "model_args": {"hidden_size": 64, "output_size": 2}
    },
    {
        "name": "4L cos_anl",
        "path": "./tb_logs/improved_model/19-15_cos.ckpt",
        "color": "green",
        "linestyle": "-.",
        "model_class": mtca.ImprovedMLPWithCosineAnnealing,
        "model_args": {"hidden_size": 128, "output_size": 2}
    },
    {
        "name": "3L simple cos_anl",
        "path": "./tb_logs/simple_mlp_zprime_new_cos_anl/version_1/checkpoints/epoch=49-step=11750.ckpt",
        "color": "red",
        "linestyle": ":",
        "model_class": SimpleMLP,
        "model_args": {"hidden_size": 64, "output_size": 2}
    }
    # 可以继续添加更多模型
]

# 创建全局字典存储所有模型的ROC指标
model_metrics = {}

# 数据预处理
df_processed = mti.preprocess_data(df, input_columns, target_column)
trans_columns = [c + '_trans' for c in input_columns]

# 分割数据集
_, test_df = train_test_split(
    df_processed,
    test_size=0.2,
    random_state=42,
    stratify=df_processed[target_column]
)

# 创建测试集数据集
test_dataset = mti.DataFrameDataset(test_df, trans_columns, target_column)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# 获取真实标签
all_labels = []
with torch.no_grad():
    for _, labels in test_loader:
        all_labels.extend(labels.numpy())
all_labels = np.array(all_labels)

# 创建图形
plt.figure(figsize=(10, 8))

# 为每个模型绘制ROC曲线
for model_info in models_info:
    model_path = model_info["path"]

    # 首先检查检查点文件是否存在
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        continue

    # 加载检查点以获取模型参数信息
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        # 处理不同的checkpoint格式
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # 处理键名不匹配问题（前缀差异）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                k = k[6:]  # 移除"model."前缀
            new_state_dict[k] = v

        # 确定输入特征大小
        if "layer1.weight" in new_state_dict:
            input_size = new_state_dict["layer1.weight"].shape[1]
            print(f"模型 {model_info['name']} 的输入特征大小: {input_size}")
        else:
            print(f"无法确定模型 {model_info['name']} 的输入特征大小，使用默认值")
            input_size = len(input_columns)

        # 根据检测到的输入大小创建模型
        model_class = model_info["model_class"]
        model_args = model_info["model_args"].copy()
        model_args["input_size"] = input_size

        model = model_class(**model_args)

        # 尝试加载state_dict
        try:
            model.load_state_dict(new_state_dict)
            print(f"成功加载模型 {model_info['name']}")
        except Exception as e:
            print(f"直接加载失败: {e}")
            # 如果直接加载失败，尝试部分加载
            model_state_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_state_dict}
            print(f"加载了 {len(pretrained_dict)}/{len(model_state_dict)} 个参数")
            model.load_state_dict(pretrained_dict, strict=False)

    except Exception as e:
        print(f"加载模型 {model_info['name']} 失败: {e}")
        continue

    model.eval()

    # 使用测试集进行预测
    all_probs = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            # 确保输入维度正确
            if inputs.shape[1] != input_size:
                # 如果输入维度不匹配，进行截断或填充
                if inputs.shape[1] > input_size:
                    inputs = inputs[:, :input_size]
                else:
                    pad = torch.zeros((inputs.shape[0], input_size - inputs.shape[1]))
                    inputs = torch.cat([inputs, pad], dim=1)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            pos_probs = probs[:, 1].numpy()
            all_probs.extend(pos_probs)

    all_probs = np.array(all_probs)

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # 保存当前模型的ROC指标到全局字典
    model_name = model_info["name"]
    model_metrics[model_name] = {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "roc_auc": roc_auc
    }

    # 打印当前模型的AUC值
    print(f"模型 {model_name} 的AUC值: {roc_auc:.4f}")

    # 绘制ROC曲线
    plt.semilogy(tpr, fpr, 
                color=model_info["color"], 
                linestyle=model_info["linestyle"],
                lw=2,
                label=f'{model_info["name"]} (AUC = {roc_auc:.4f})')



    # 生成随机分类器的点
    random_fpr = np.logspace(-5, 0, 100)  # 在对数空间中均匀采样
    random_tpr = random_fpr  # 随机分类器满足 TPR = FPR
    plt.semilogy(random_tpr, random_fpr, 'k--', lw=2, label='Random')

# 设置图形属性
plt.xlim([1e-5, 1.0])  # 设置x轴范围，避免0值
plt.ylim([1e-5, 1.0])  # 设置y轴范围，避免0值
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('log scale ROC curves', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
# plt.grid(True, which="both", ls="--", alpha=0.7)
plt.tight_layout()
plt.savefig('multi_model_roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印所有模型的ROC指标
print("\n所有模型的ROC指标:")
for model_name, metrics in model_metrics.items():
    print(f"模型: {model_name}")
    print(f"  AUC: {metrics['roc_auc']:.4f}")
    print(f"  FPR长度: {len(metrics['fpr'])}")
    print(f"  TPR长度: {len(metrics['tpr'])}")
    print(f"  阈值数量: {len(metrics['thresholds'])}")

# 现在可以通过model_metrics字典访问任何模型的ROC指标
# 例如: model_metrics['4L simple cos_anl']['fpr']
#       model_metrics['4L simple cos_anl']['tpr']
#       model_metrics['4L simple cos_anl']['roc_auc']
