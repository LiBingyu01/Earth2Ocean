<div align="center">

<h1>Exploring the Underwater World Segmentation without Extra Training</h1>

[Paper](https://arxiv.org/pdf/2511.07923) &nbsp;&nbsp;&nbsp;&nbsp; [Open-sourced Datasets](https://1drv.ms/f/c/69a773fee5342110/Eo2k4_Rxk4xLvYnLgP7YVccBytHqIKTqheVp1cKmZ8XXtw?e=beZeh6)

<img src="assets/Earth2Ocean_Fun_Guide.png" width="700px"/>

<img src="assets/Earth2Ocean.gif" width="700px"/>

</div>

## Abstract
> Accurate segmentation of marine organisms is vital for biodiversity monitoring and ecological assessment, yet existing datasets and models remain largely limited to terrestrial scenes. To bridge this gap, we introduce **AquaOV255**, the first large-scale and fine-grained underwater segmentation dataset containing 255 categories and over 20K images, covering diverse categories for open-vocabulary(OV) evaluation. Furthermore, we establish the first underwater OV segmentation benchmark, **UOVSBench**, by integrating AquaOV255 with five additional underwater datasets to enable comprehensive evaluation. Alongside, we present **Earth2Ocean**, a training-free OV segmentation framework that transfers terrestrial vision–language models (VLMs) to underwater domains without any additional underwater training. Earth2Ocean consists of two core components: a Geometric-guided Visual Mask Generator (**GMG**) that refines visual features via self-similarity geometric priors for local structure perception, and a Category-visual Semantic Alignment (**CSA**) module that enhances text embeddings through multimodal large language model reasoning and scene-aware template construction. Extensive experiments on the UOVSBench benchmark demonstrate that Earth2Ocean achieves over significant performance improvement on average while maintaining efficient inference. The clear code is provided in the appendix.*

## Dependencies and Installation


```
# create new anaconda env
conda create -n Earth2Ocean python=3.10
conda activate Earth2Ocean

# install torch and dependencies
pip install -r requirements.txt
```

## Datasets
We include the following dataset configurations, you can download the dataset from [OneDrive](https://1drv.ms/f/c/69a773fee5342110/Eo2k4_Rxk4xLvYnLgP7YVccBytHqIKTqheVp1cKmZ8XXtw?e=beZeh6) and [BaiDu Disk](https://pan.baidu.com/s/1gg1vLA9AICISjOksge2v8g?pwd=USTC): 
```
AquaOV255, dutuseg, mas3k, SUIM, USIS10K, usis16k.
```
I have find the errors.
For AquaOV255, please delete *Catfish_112.png* and use the following scrpt to relabel some masks.
```
import numpy as np
from PIL import Image
import os


paths = [
    "datasets/AquaOV255/masks/Lanternfish_003.png",
    "datasets/AquaOV255/masks/Lanternfish_001.png",
    "datasets/AquaOV255/masks/Lanternfish_002.png",
]

save_dir = "datasets/AquaOV255/masks"
os.makedirs(save_dir, exist_ok=True)

for path in paths:
    mask = np.array(Image.open(path))

    if mask.ndim != 2:
        raise ValueError(f"Not single-channel mask: {path}, shape={mask.shape}")

    mask = mask.astype(np.uint16)
    mask[mask == 254] = 206

    filename = os.path.basename(path)
    save_path = os.path.join(save_dir, filename)

    Image.fromarray(mask, mode="I;16").save(save_path)
    print(f"[SAVED] {save_path}")
```


## Model evaluation
Please modify some settings in `configs/base_config.py` before running the evaluation.

For **DepthAnything Model** and **Open-CLIP Model** please download from their official website or our [LINK](https://1drv.ms/f/c/69a773fee5342110/Es-h3W58AfVKkaqbgZHHxtoB97YPTtrl2lWUAw_64kma5A?e=Hk9Puc) and put them in `./pretrained_ckpt/`.

Evaluation on all datasets:
```
sh run.sh
```

The results will be saved in `./work_logs/`.


## Cite our paper
```
@article{li2025exploring,
  title={Exploring the Underwater World Segmentation without Extra Training},
  author={Li, Bingyu and Huo, Tao and Zhang, Da and Zhao, Zhiyuan and Gao, Junyu and Li, Xuelong},
  journal={arXiv preprint arXiv:2511.07923},
  year={2025}
}
```
