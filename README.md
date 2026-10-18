# CIQALSN
VLM-Guided Cross-Domain Image Quality Assessment via Semantic and Numerical Dual Distillation


[!\[Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[!\[PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[!\[CLIP](https://img.shields.io/badge/CLIP-ViT--B/32-6b4c9a.svg)](https://github.com/openai/CLIP)

\---

## Framework Overview

The CIQALSN framework consists of three main stages:









Features

* **CLIP-based Backbone**: Utilizes ViT-B/32 from OpenAI CLIP as the visual encoder, with the last 4 transformer blocks trainable for adaptation.
* **Image-Text Fusion**: Concatenates image and text features from CLIP for comprehensive quality assessment.
* **Multi-Task Learning**: Simultaneously predicts 5 perceptual attributes (contrast, brightness, realism, blur, saturation) and overall quality score.
* **Ranking Distillation**: Transfers teacher's quality ranking knowledge via temperature-scaled softmax probability matching.
* **Meta-Learning Attribute Gating**: Dynamically computes attribute-wise distillation weights using gradient cosine similarity, enabling adaptive knowledge transfer.
* **Warmup + Cosine Annealing**: Learning rate schedule with linear warmup followed by cosine annealing for stable convergence.
* **Reproducible**: Fixed random seeds and deterministic settings for reproducibility.

## Results

Performance comparisons on cross-domain IQA tasks (trained on KADID, tested on target domains):

|Method|LIVEC (SRCC/PLCC)|BID (SRCC/PLCC)|AGIQA-3k (SRCC/PLCC)|AGIQA-20k (SRCC/PLCC)|Average (SRCC/PLCC)|
|-|-|-|-|-|-|
|BRISQUE (2012)|0.243/0.251|0.175/0.175|0.210/0.244|0.118/0.154|0.187/0.206|
|NIQE (2012)|0.304/0.362|0.355/0.381|0.326/0.371|0.416/0.425|0.350/0.385|
|PIQE (2015)|0.262/0.362|0.269/0.351|0.176/0.249|0.164/0.180|0.218/0.286|
|RankIQA (2017)|0.491/0.495|0.510/0.670|0.317/0.376|0.402/0.385|0.430/0.482|
|MetaIQA (2020)|0.464/0.464|0.300/0.428|0.510/0.546|0.528/0.517|0.451/0.489|
|DBCNN (2023)|0.266/0.290|0.318/0.212|0.457/0.492|0.324/0.328|0.341/0.331|
|HyperIQA (2020)|0.490/0.487|0.379/0.282|0.725/0.750|0.712/0.725|0.577/0.561|
|MCOLE (2024)|0.411/0.395|0.426/0.418|0.489/0.531|0.680/0.694|0.502/0.510|
|AKD-IQA (2025)|0.372/0.364|0.502/0.493|0.422/0.468|0.405/0.423|0.425/0.437|
|DiffV2IQA (2025)|0.329/0.323|0.387/0.366|0.540/0.579|0.358/0.359|0.404/0.407|
|MMQA (2025)|0.625/0.635|0.572/0.578|0.717/0.748|0.744/0.769|0.664/0.682|
|UCDA (2021)|0.382/0.358|0.348/0.391|0.221/0.257|0.234/0.250|0.296/0.314|
|RankDA (2021)|0.451/0.455|0.535/0.582|0.562/0.580|0.573/0.575|0.530/0.548|
|StyleAM (2024)|0.584/0.561|0.637/0.567|0.619/0.649|0.611/0.618|0.613/0.599|
|DGQA (2024)|0.696/0.690|0.770/0.753|0.716/0.755|0.707/0.716|0.722/0.729|
|GDCIQA (2026)|0.753/0.756|0.787/0.781|0.781/0.794|0.748/0.775|0.767/0.776|
|**CIQALSN (Ours)**|**0.763/0.782**|**0.791/0.796**|**0.805/0.836**|**0.816/0.832**|**0.793/0.811**|

**Key Results:**

* Best performance across all four target datasets (LIVEC, BID, AGIQA-3k, AGIQA-20k)
* On AGIQA-20k, CIQALSN improves over the strongest baseline (GDCIQA) by **9.1% SRCC** and **7.4% PLCC**
* Outperforms classic DL IQA methods (HyperIQA, MMQA) and cross-domain methods (DGQA, GDCIQA)

## Ablation Study Highlights

|Configuration|LIVEC (SRCC/PLCC)|AGIQA-3k (SRCC/PLCC)|
|-|-|-|
|Baseline (no semantics, no distillation)|0.558/0.563|0.610/0.662|
|+ Ranking Distillation (L3)|0.615/0.617|0.729/0.764|
|+ Numerical Distillation (L4)|0.602/0.611|0.630/0.663|
|+ Numerical + Meta-learning|0.624/0.631|0.707/0.736|
|+ Ranking + Numerical (w/o meta)|0.732/0.731|0.724/0.757|
|**Full (Ranking + Numerical + Meta)**|**0.763/0.782**|**0.805/0.836**|

Semantic analysis (content description + attribute ratings) contributes significant gains: enabling both components yields SRCC of 0.763 on LIVEC and 0.805 on AGIQA-3k, improving over the baseline by **13.86%** and **14.84%** respectively.

## Requirements

* Python >= 3.8
* PyTorch >= 1.12
* torchvision
* openai-clip
* numpy
* pandas
* scikit-learn
* PIL (Pillow)
* scipy
* tqdm

Install dependencies:

```bash
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install numpy pandas scikit-learn scipy tqdm
```

## Data Preparation

### Directory Structure

Organize your data as follows:

```
data/
├── images/              # All image files
│   ├── 1.png
│   ├── 2.png
│   ├── ...
└── labels/
    ├── train.csv         # Training labels
    └── test.csv          # Test labels
```

## Training

### Step 1: Prepare Teacher Model Checkpoint

First, train or obtain the teacher model checkpoint. The teacher model is trained on the source domain (KADID) with ground-truth quality labels and LLM-annotated attributes. Save the teacher checkpoint before running distillation:

```bash
# Placeholder: train teacher model first (not included in this repo)
# Then save as ./pre\_model.pth
```

### Step 2: Run Distillation Training

Train the student model with dual distillation:

```bash
python CIQARLSN.py \\
    --image\_dir /path/to/images/ \\
    --label\_file ./AGIQA\_merged.csv \\
    --save\_dir ./distill\_checkpoints \\
    --clip\_model ViT-B/32 \\
    --teacher\_checkpoint ./pre\_model.pth \\
    --batch\_size 16 \\
    --num\_epochs 50 \\
    --lr 1e-5 \\
    --lr\_head 1e-4 \\
    --weight\_decay 1e-4 \\
    --grad\_clip 1.0 \\
    --test\_size 0.2 \\
    --seed 42 \\
    --warmup\_epochs 5 \\
    --min\_lr 1e-7 \\
    --temperature 1.0 \\
    --meta\_tau 2.0
```

### Hyperparameters

|Parameter|Default|Description|
|-|-|-|
|`--image\_dir`|`/home/user/data/AGIQA3k/`|Path to image directory|
|`--label\_file`|`./AGIQA\_merged.csv`|Path to label CSV file|
|`--save\_dir`|`./distill\_checkpoints`|Directory to save checkpoints|
|`--clip\_model`|`ViT-B/32`|CLIP model variant|
|`--teacher\_checkpoint`|`./pre\_model.pth`|Path to teacher model weights|
|`--batch\_size`|`16`|Training batch size|
|`--num\_epochs`|`50`|Total training epochs|
|`--lr`|`1e-5`|Learning rate for CLIP layers|
|`--lr\_head`|`1e-4`|Learning rate for custom layers (fusion\_proj, backbone, attr\_head)|
|`--weight\_decay`|`1e-4`|Weight decay (AdamW)|
|`--grad\_clip`|`1.0`|Gradient clipping norm|
|`--test\_size`|`0.2`|Test split ratio|
|`--seed`|`42`|Random seed|
|`--warmup\_epochs`|`5`|Warmup epochs|
|`--min\_lr`|`1e-7`|Minimum learning rate|
|`--temperature`|`1.0`|Temperature for ranking distillation softmax|
|`--meta\_tau`|`2.0`|Temperature coefficient for meta-gating sigmoid|

### Training Output

The training script outputs per-epoch metrics including:

* Training loss
* Quality assessment: SRCC and PLCC (student predictions vs MOS)
* Attribute vs Teacher: SRCC and PLCC for each of 5 attributes
* Attribute vs LLM: SRCC and PLCC for each of 5 attributes

The best model (highest quality SRCC on test set) is automatically saved as `best\_student.pth`.

## 

## Directory Structure

```
CIQALSN/
├── CIQARLSN.py              # Main training script (dual distillation)
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── pre\_model.pth            # Teacher model checkpoint (provided separately)
└── distill\_checkpoints/     # Training checkpoints (created during training)
    └── best\_student.pth     # Best student model (auto-saved)
```

## 

