# BIE-EdgeNet
BIE-EdgeNet: Difference-Prior Cross-Attention and Edge Fusion for Building Change Detection in High-Resolution Imagery
# Requirement
- Python 3.8
- Pytorch 1.10.1
- timm 0.4.12
- einops
- opencv-python (cv2)
- tqdm
# Dataset
The dataset should consist of bi-temporal image pairs with pixel-level change labels and edge labels. Each sample provides the pre-event image, the post-event image, the binary change mask, and the corresponding edge mask.

Organize the dataset into the following structure:
```
/E:/Data_AI
  /CDCD
    /A
    /B
    /label
    /label_edge
    /list
        train.txt
        val.txt
        test.txt
  /LEVIR-CD
    ...
  /WHU-CD
    ...
```
Where folder `A` contains pre-event images, folder `B` contains post-event images, folder `label` contains the binary change masks (0/255), and folder `label_edge` contains the edge masks generated from `label`.

The edge labels can be generated from the binary change masks via:
```
python edge_making.py
```
Then register your dataset path in `data_config.py`, e.g. `root_dir = r'E:\Data_AI\LEVIR-CD'`.
# Working Example
1. Training the Model.

(1) Prepare the Dataset.
Make sure your dataset is structured as described above and adjust the `--data_name` and `--project_name` arguments (or the default values in `main_cd.py`) to match your dataset.
(2) Training Script.
To train the BIE-EdgeNet model, run the following command:
```
python main_cd.py
```
Note: The model parameters are set to `batch_size` of 16, `epochs` of 200, and `learning_rate` of 1e-4 (AdamW), which you can modify according to your needs.
(3) Model Output.
At the end of training, the model's parameters will be saved in the `./checkpoints/<project_name>/` folder (including `best_ckpt.pt` and `last_ckpt.pt`). You can use the saved model for inference.

2. Testing the Model.

(1) Prepare the Dataset.
Make sure your dataset is structured as described above and adjust the `--data_name` argument in the testing script to point to your testing data.
(2) Testing Script.
To evaluate the trained model on the test dataset, run the following command:
```
python eval_cd.py
```
Note: The `eval_cd.py` script loads the pre-trained model from `./checkpoints/<project_name>/best_ckpt.pt`. You can select the appropriate pre-trained model parameters as input.
(3) Model Output.
The model's prediction maps and evaluation metrics (F1, IoU, OA) are saved in the `./vis/<project_name>/` folder.

Additional Note: Ablation studies can be performed by switching the `mode` argument (`ALL`, `CNN_Tr`, `CNN_Tr_BIE`, `CNN_Tr_Edge`, `Edge`) in `main_cd.py`.
# Acknowledgments
The code of "ResNet34" is based upon [torchvision](https://github.com/pytorch/vision).
The comparison experiments reuse the open-source implementations of [ChangeFormer](https://github.com/wgcban/ChangeFormer), [ChangeMamba](https://github.com//ChenHongruixuan/MambaBCD), [ChangeDINO](https://github.com//ChenHongruixuan/ChangeDINO), [B2CNet](https://github.com//Z-ZX-B/CDBaseline), and [BIT](https://github.com//justchenhao/BIT_CD).
Thanks for their excellent works!
# Citation
If you use this code for your research, please cite our paper.
```
@article{BIE-EdgeNet,
  title = {A Bi-directional Information Exchange and Edge-aware Network for Change Detection in High-Resolution Remote Sensing Images},
  author = {...},
  journal = {...},
  year = {202X},
}
```
