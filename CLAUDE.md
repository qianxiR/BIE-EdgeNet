# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BIE-EdgeNet is a PyTorch research codebase for **bi-temporal remote sensing change detection (CD)**. The core contribution is the `BIE_EdgeNet` model, which combines CNN features, Transformer features, and a Bi-directional Information Exchange (BIE) module with edge-aware fusion. The repo also vendored several SOTA baselines (ChangeFormer, ChangeMamba, ChangeDINO, B2CNet, EGPNet, EGRCNN, HSANet, AERNet, etc.) for ablation/comparison.

The project is a **single-gpu Python script** project (no package install / no test suite). It is run via `python main_cd.py`.

## Environment

- Python 3.8 + PyTorch 1.10.1 (CUDA 10.2). See `requirements.txt` (conda env spec). Key deps: `torchvision`, `timm==0.4.12`, `einops`, `opencv-python`, `gdal` (osgeo), `tqdm`, `matplotlib`.
- GDAL is required only by `models/evaluator.py` (large GeoTIFF inference path) and `tool/img_concat_geo.py`.
- Windows paths are hardcoded throughout `data_config.py`; expect to edit them per machine.

## Common Commands

Run training + eval (single command, sequential `train(...)` then `test(...)`):
```bash
python main_cd.py
```

Evaluation-only entry point (separate parser, uses `models.evaluator.CDEvaluator`):
```bash
python eval_cd.py
```

Quick-start inference demo using sample images in `samples_LEVIR/`:
```bash
python demo_LEVIR.py
```

Generate edge labels from binary change masks (morphological gradient, required by `CDDataset` — the dataset expects a `label_edge/` folder alongside `label/`):
```bash
python edge_making.py            # edit the hardcoded label_dir / save_dir first
```

There is **no build step, no lint config, and no automated tests**. `.pytest_cache/` exists but no test files do.

## How to Run a Different Experiment

`main_cd.py` is structured as multiple sequential `ArgumentParser` blocks. To switch experiments you **edit the file** rather than pass CLI flags:
- Comment/uncomment the `train(...)` / `test(...)` calls at the bottom.
- Edit `--net_G` (model registry key — see `models/networks.py:define_G`), `--data_name` (key into `data_config.DataConfig`), `--loss`, `--project_name` (names the checkpoint/vis subfolder).
- Switch between `CDTrainer` and `CDTrainer_fp16` (mixed-precision) in `train()` by uncommenting the desired line.

## Architecture

### Entry flow
```
main_cd.py
  └── utils.get_loaders(args)            # builds train/val DataLoaders
        └── datasets/CD_dataset.py:CDDataset   (reads A/, B/, label/, label_edge/, list/{train,val,test}.txt)
  └── models/trainer.py:CDTrainer        # OR trainer_FP16.py:CDTrainer_fp16
        ├── models/networks.py:define_G  # model factory dispatch on args.net_G
        ├── models/losses.py             # loss factory dispatch on args.loss / args.net_G
        └── misc/metric_tool.py:ConfuseMatrixMeter  # accumulates confusion matrix → F1/IoU/Acc
```

### Model factory — `models/networks.py:define_G`
Single dispatch point. Adding a new architecture means: (1) write the model class, (2) add an `elif args.net_G == 'YOUR_KEY':` branch instantiating it, (3) if it needs a custom loss or post-processing, also wire it into the trainer/evaluator dispatch tables.

### Trainer dispatch tables — `models/trainer.py`
`CDTrainer` has multiple per-model branches that must be extended in lockstep when adding a model:
- `__init__`: loss selection (`self._pxl_loss`)
- `_forward_pass`: how `self.G_pred` (and optional `self.multi_scale_preds`, `self.Edge_pred`, `self.out_list`, `self.preds`, etc.) are extracted from `net_G` output
- `_update_metric`: per-model prediction decoding (argmax vs. sigmoid vs. `[:, 1, :, :]` + threshold)
- `_backward_G`: per-model loss aggregation

**If a new model's forward output shape/structure differs, all four branches need a case.** Otherwise metrics will be silently wrong.

### Evaluator — `models/evaluator.py:CDEvaluator`
Mirrors the trainer's per-model branches for inference decoding + visualization. Also implements:
- `eval_models(checkpoint_name, mode=...)` — the ablation `mode` kwarg toggles sub-modules on/off **specifically for `BIE_EdgeNet`** via `net_G.FE_IMD.set_test_mode('CNN'|'Tr'|'BIE'|'Edge_Fusion', bool)` and `net_G.CD_ED.set_Edge_mode(bool)`. Valid modes: `'ALL'`, `'CNN_Tr'`, `'CNN_Tr_BIE'`, `'CNN_Tr_Edge'`, `'Edge'`. **Calling these setters on any other model will raise AttributeError.**
- `predict_large_image` / `pred_gdal_blocks_write` — tiled GeoTIFF inference using GDAL.

### BIE_EdgeNet internals — `models/BIE_EdgeNet.py` + `models/BIE_Cross_Attentions.py`
- Sub-module `FE_IMD` (Feature-Extract / Information-Mix-Diff) exposes `set_test_mode(component, bool)` for ablation. Components: `CNN`, `Tr` (Transformer), `BIE`, `Edge_Fusion`.
- Sub-module `CD_ED` exposes `set_Edge_mode(bool)` to toggle edge-aware fusion in the change decoder.
- `FourStage_Diff_Enhancer` applies OCDA (overlap cross attention) at stages 1–3 and CDA at stage 4.

### Datasets — `datasets/CD_dataset.py`
Expected on-disk layout per dataset root:
```
<A/>            pre-event images
<B/>            post-event images
<label/>        binary change masks (PNG)
<label_edge/>   edge masks (PNG) — generated by edge_making.py
<list>/<train|val|test>.txt   filename lists
```
Labels are binarized: any non-zero pixel → 255 → then normalized to 1. `L` and `L_edge` are returned in each batch dict.

### Data config — `data_config.py`
Hardcoded `data_name → root_dir` mapping (Windows paths). To add a dataset, add an `elif` branch in `get_data_config` and point `root_dir` at your local copy. Datasets referenced include LEVIR-CD, WHU-CD, and CDCD at multiple crop sizes (256/512/1024) and train/val/test splits (e.g. `7_1_2`, `7_2_1`).

## Key Conventions

- **`self.G_pred` is overloaded.** Depending on `--net_G`, it may be a single tensor, a list of multi-scale preds (with `[-1]` being the final-resolution prediction), or a tuple. Always check the corresponding `_forward_pass` branch before assuming shape.
- **Loss dispatch is dual-keyed.** Some losses are selected by `args.loss` (`ce`, `bce`, `fl`, `miou`, `mmiou`, `eas`, `BCEDiceLoss`, `RCDT_MultiScale_Loss`, `AERNet_Loss`); others by `args.net_G` (`HSANet`, `LENet`, `B2CNet`, `ChangeDINO`, `ChangeMamba`, `EGRCNN`, `EGPNet`, `EATDer`, `BGSNet`). The `args.net_G` branch wins if the model has a bespoke loss.
- **Checkpoints** are written to `checkpoints/<project_name>/`; visualizations to `vis/<project_name>/`. `best_ckpt.pt`, `best_acc_ckpt.pt`, `last_ckpt.pt` are the conventional names loaded in `eval_models`.
- **`indice0.txt` / `indice1.txt`** (≈14 MB each) at repo root are large index files used by some dataset preprocessing — not source code.
- **`replay_pid*.log`** files are auto-generated MATLAB/Python crash replays and can be ignored/deleted.
- Comments and docstrings are largely in Chinese.

## Vendored Baselines (do not edit unless explicitly asked)

These subdirectories contain third-party SOTA implementations used for comparison. They are imported by `models/networks.py` via absolute paths and should not be refactored without checking the import graph:
- `models/ChangeDINO/`, `models/ChangeMambaBCD/`, `models/AERNet/`, `models/EGPNet/`, `models/EGRCNN/`, `models/lenet_master/` (PaddleSeg-style segmentation toolkit), `B2CNetmain/`, `BMCNet-ESR-master/`, `lenet-master/`, `dinov3/`.

`models/lenet_master/` in particular is a large vendored segmentation framework; its `docs/`, `projects/`, and `paddleseg/` tree is unrelated to the BIE-EdgeNet training loop and only the specific model files imported from it are used.
