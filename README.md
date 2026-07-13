# BIE-EdgeNet:面向双时相遥感图像变化检测的双向信息交换与边缘感知网络

本仓库是论文 *"BIE-EdgeNet: A Bi-directional Information Exchange and Edge-aware Network for Change Detection in Remote Sensing Images"* 的官方 PyTorch 实现。

## 目录

- 数据集准备
- 训练
- 评估
- 推理
- 引用
- 致谢

## 数据集准备

下载 LEVIR-CD / WHU-CD / CDCD 数据集,并按如下结构组织(三个数据集共用同一父目录 `E:\Data_AI\`):

```
E:\Data_AI\
├─CDCD
│  ├─A                  # T1 时相影像
│  ├─B                  # T2 时相影像
│  ├─label              # 二值变化标签 (0/255)
│  ├─label_edge         # 边缘标签(由 label 生成)
│  └─list
│      ├─train.txt
│      ├─val.txt
│      └─test.txt
├─LEVIR-CD
│  ├─A
│  ├─B
│  ├─label
│  ├─label_edge
│  └─list
└─WHU-CD
    ├─A
    ├─B
    ├─label
    ├─label_edge
    └─list
```

其中 `label_edge/` 由 `label/` 生成:

```bash
python edge_making.py
```

数据集路径在 `data_config.py` 中注册,例如:

```python
elif data_name == 'LEVIR-256-edge':
    self.label_transform = "norm"
    self.root_dir = r'E:\Data_AI\LEVIR-CD'
```

## 训练

BIE-EdgeNet 的总体架构:

![architecture](resource/architecture.png)

### 环境安装

**Step 1**:创建 conda 环境并激活。

```bash
conda create -n bieedgenet python=3.8
conda activate bieedgenet
```

**Step 2**:克隆仓库。

```bash
git clone https://github.com/<your-account>/BIE-EdgeNet.git
cd ./BIE-EdgeNet
```

**Step 3**:安装依赖。

```bash
pip install -r requirements.txt
```

### 开始训练

```bash
python main_cd.py
```

默认配置:`img_size=256, batch_size=16, lr=1e-4, AdamW, 200 epochs, loss=eas`。切换数据集或损失函数请编辑 `main_cd.py` 末尾的 `ArgumentParser` 块。

## 评估

```bash
python eval_cd.py --data_name LEVIR-256-edge --net_G BIE_EdgeNet --checkpoint_name best_ckpt.pt
```

消融研究可在 `main_cd.py` 的 `test(args, mode=...)` 中切换 `mode`,可选值:`ALL` / `CNN_Tr` / `CNN_Tr_BIE` / `CNN_Tr_Edge` / `Edge`。

## 推理

```bash
python demo_LEVIR.py
```

输入影像对位于 `samples_LEVIR/`,预测结果保存至 `samples_LEVIR/predict/`。

## 引用

如果您觉得本仓库对您的研究有帮助,请考虑引用:

```bibtex
@article{BIEEdgeNet202X,
  title   = {BIE-EdgeNet: A Bi-directional Information Exchange and Edge-aware Network for Change Detection in Remote Sensing Images},
  author  = {...},
  journal = {...},
  year    = {202X},
  volume  = {},
  number  = {},
  pages   = {},
  doi     = {}
}
```

## 致谢

感谢以下开源仓库:ChangeFormer、ChangeMamba、ChangeDINO、B2CNet、BIT。

## License

本仓库遵循 MIT License,代码仅用于学术研究。

## 联系方式

如有任何问题,欢迎提 issue 或联系 `<your-email>`。
