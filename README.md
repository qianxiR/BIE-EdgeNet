# BIE-EdgeNet
Difference-Verified Bilateral Interaction and Edge Restoration for Building Change Detection in High-Resolution Remote Sensing Imagery

# Requirement
- Python 3.8
- Pytorch 1.10.1
- timm 0.4.12
- einops
- opencv-python (cv2)
- tqdm

# Dataset
The dataset should consist of bi-temporal image pairs with pixel-level change labels and edge labels. Each sample provides the pre-event image, the post-event image, the binary change mask (label), and the corresponding edge mask (label_edge).

For building change detection (BCD), download the [LEVIR-CD](https://justchenhao.github.io/LEVIR/), [WHU-CD](http://gpcv.whu.edu.cn/data/building_dataset.html). Crop each image into 256x256 patches and prepare the dataset into the following structure.

An example sample of the CDCD dataset, "cdcd_demo.7z", can be downloaded from the [Release](https://github.com/qianxiR/BIE-EdgeNet/releases) page and unzipped for format reference.

Organize the dataset into the following structure:

```
/E:/Data_AI
  /CDCD
    /A                  pre-event images
    /B                  post-event images
    /label              binary change masks (0/255)
    /label_edge         edge masks generated from labels
    /list
      train.txt
      val.txt
      test.txt
  /LEVIR-CD
    ...
  /WHU-CD
    ...
```

The edge masks in `label_edge/` can be generated from the binary change masks via:
```
python edge_making.py
```
Then register your dataset path in `data_config.py`, e.g. `root_dir = r'E:\Data_AI\LEVIR-CD'`.

# Working Example

0. Environment
```bash
cd BIE-EdgeNet
conda create -n BIE python=3.8
conda activate BIE
# PyTorch 1.10.1 + CUDA 10.2
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=10.2 -c pytorch
pip install timm==0.4.12 einops opencv-python tqdm
```

1. Training the Model.

(1) Prepare the Dataset. Make sure your dataset is structured as described above and adjust the `--data_name` and `--project_name` arguments in `main_cd.py` to point to your dataset. (2) Training Script. To train the BIE-EdgeNet model, run the following command:
```
python main_cd.py
```
Note: The model parameters are set to batch_size of 16, epochs of 200, and learning_rate of 1e-4 (AdamW), which you can modify according to your needs. (3) Model Output. At the end of training, the model's parameters will be saved in the `./checkpoints/<project_name>/` folder, including `best_ckpt.pt` and `last_ckpt.pt`. You can use the saved model for inference.

2. Testing the Model.

(1) Prepare the Dataset. Make sure your dataset is structured as described above and adjust the `--data_name` argument in the testing script to point to your testing data folder. (2) Testing Script. To test the trained model on the test dataset, run the following command:
```
python eval_cd.py
```
Note: The `eval_cd.py` script loads the pre-trained model from `./checkpoints/<project_name>/best_ckpt.pt`. You can select the appropriate pre-trained model parameters as input. (3) Model Output. The model's prediction maps and evaluation metrics (F1, IoU, OA) are saved in the `./vis/<project_name>/` folder.

Additional Note: Ablation studies can be performed by switching the `mode` argument in `main_cd.py` (`ALL`, `CNN_Tr`, `CNN_Tr_BIE`, `CNN_Tr_Edge`, `Edge`).

# Acknowledgments

The  dataset is constructed based on the LEVIR-CD and WHU-CD change detection datasets. Thanks for their excellent works!
```
@article{chen2020spatial,
  title={A spatial-temporal attention-based method and a new dataset for remote sensing image change detection},
  author={Chen, Hao and Shi, Zhenwei},
  journal={Remote Sensing},
  volume={12},
  number={10},
  pages={1662},
  year={2020},
  publisher={MDPI}
}
```

```
@article{ji2018fully,
  title={Fully convolutional networks for multisource building extraction from an open aerial and satellite imagery data set},
  author={Ji, Shunping and Wei, Shiqing and Lu, Meng},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={57},
  number={1},
  pages={574--586},
  year={2018},
  publisher={IEEE}
}
```

# Citation
If you use this code for your research, please cite our paper.

```
@article{BIE-EdgeNet,
  title={BIE-EdgeNet: Difference-Prior Cross Attention and Edge Fusion for Building Change Detection in High-Resolution Imagery},
  author={Qiao Zhang and Yuhuan Zhang and Guangliang Cheng and Zhoufeng Wang and Qianxi Rong and Yang Shen and Xuxiong Xu},
  journal={...},
  year={202X},
}
```
