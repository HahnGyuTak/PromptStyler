# PromptStyler

A Pytorch implementation of [PromptStyler](https://arxiv.org/abs/2307.15199)

## Training

### Dataset List

* PACS
* VLCS
* OfficeHome

### Shell script
```Shell
CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset PACS --train both --infer both
```

### Train Step
1. Train style prompt (`--train style`)
2. Visualization style-content vector with Diffuser (`--infer diff`)
    * Save path : `result/{--dataset}/txt2img_res_f32`
3. Train Classifier with prompt and images (`--train classifier`)
4. Inference with Real Dataset which you used (`--infer classifier`)
    * Real Dataset path : `/home/dataset/`


### Options

#### Required
* `--dataset` : Which dataset to use
* `--train` : Select 1 of [`style`, `classifier`, `both`]
* `--inference` : Select 1 of [`diff`, `classifier`, `both`]

#### Selection

* `--config` : To use your config file, enter the `path` (See [this]((config/config_k80.yaml)) for the config file format.)
  ```Shell
  CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset OfficeHome_K200 --train both --infer both --config config/config_vitb32_K200.yaml
  ```

