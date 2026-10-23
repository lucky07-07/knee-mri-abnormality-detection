# Knee MRI Abnormality Detection

**Multi-label detection of 12 knee abnormalities from multi-series MRI, trained on labels extracted from multilingual radiology reports with an LLM.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/timm-ConvNeXt-blue)
![Transformers](https://img.shields.io/badge/HF%20Transformers-Qwen2.5--14B-FFD21E?logo=huggingface&logoColor=black)
![Kaggle](https://img.shields.io/badge/Kaggle-RSNA%20Knee%202026-20BEFF?logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

![Pipeline](docs/pipeline.png)

## Results

| Evaluation set | Macro ROC-AUC |
|---|---|
| Kaggle **private** leaderboard (hidden test studies) | **0.906** |
| Kaggle public leaderboard | 0.920 |
| 58 radiologist-labelled studies ("gold", never trained on) | 0.907 |

*RSNA Knee Abnormality Detection, a Kaggle research code competition (6,425 teams, offline inference, 9-hour limit).*

<p align="center"><img src="reports/figures/05_per_finding_auc.png" width="49%"> <img src="reports/figures/02_label_quality.png" width="49%"></p>

## The problem

The task is to predict 12 findings for every knee MRI study: ACL, MCL, medial and lateral meniscus tears, medial, lateral and patellofemoral OA, effusion, synovitis, Baker's cyst, bone contusion and fracture. The scoring metric is macro ROC-AUC.

The hard part is the labels, not the images:

- The dataset is about **820K DICOM files (570 GB)**. Of the **4,407 training studies**, only **58** have labels read from the images by radiologists.
- The other 4,349 have only a **free-text radiology report**, written in about ten languages (English, Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch, French and others).
- The **test set has no reports**: predictions must come from the images alone.

## Approach

**1. Report labels** ([notebook 02](notebooks/02_report_labelling.ipynb), [`kneemri/labels`](src/kneemri/labels))
- A multilingual **rule-based labeler** (regex with negation handling) is the fallback source.
- **LLM grading:** Qwen2.5-14B-Instruct (4-bit, on 2× T4) grades every report 0–3 per finding as JSON. It reads the probabilities of the grade-digit tokens to get a continuous *expected grade*. This breaks ties and improves ranking quality.
- **Teacher labels:** predictions of earlier image models (distillation).
- Grades are **calibrated** to probabilities on the gold studies, with a Beta prior and a monotone constraint. Sources are then **blended per finding** with weight (gold AUC − 0.5)².

  Label quality against gold (macro AUC): rule-based **0.71** → LLM **0.88–0.89** → blend **0.93**.

**2. DICOM preprocessing** ([`kneemri/preprocessing.py`](src/kneemri/preprocessing.py), [`kneemri/cache.py`](src/kneemri/cache.py))
- Each study's series are assigned to 6 slots: sagittal, coronal and axial, each fluid-sensitive or not.
- Each series is reoriented to a canonical view using DICOM geometry, and left knees are mirrored to right.
- The image is cropped to a physical field of view, clipped to intensity percentiles, resampled to a fixed slice count, and cached as uint8.
- Three views of every study feed the ensemble: a **140 mm** field of view, a **110 mm zoom**, and the **full native** field of view.

**3. Model: KneeNet** ([`kneemri/model.py`](src/kneemri/model.py))
- Every 2.5D window (3 adjacent slices) of every series becomes one token through a shared ImageNet ConvNeXt-T.
- Slot and depth embeddings are added, then one **transformer layer** gives context across slices and series.
- **12 attention-pooling heads**, one per finding, produce the 12 logits. Missing series are masked out of the attention.
- **Training:** BCE on soft labels, EMA weights, AdamW with a lower backbone learning rate, one-cycle schedule.
  - Augmentation runs on the GPU: affine and gamma transforms, plus an *anatomical* left/right mirror that keeps medial as medial. Plain flips are never used, because they would swap medial and lateral.
  - Each training step reads only 16 random windows per study; inference reads all of them.

**4. Ensemble and inference** ([`kneemri/inference.py`](src/kneemri/inference.py))
- There are 3 views × 5 folds = **15 models**.
- Test-time augmentation averages identity with zoom, or with mirror for the mirror-trained family.
- Predictions are rank-averaged within each family, then the families are averaged with equal weight.
- The submission is validated before it is written, and the whole pipeline runs offline within Kaggle's time limit.

<p align="center"><img src="reports/figures/05_ensemble_gain.png" width="70%"></p>

## Key findings

- **Labels were the bottleneck, not the model.** The image models cannot beat their training labels by much. Every large score gain came from better report labels, not from bigger backbones or more epochs.
- **Images can still go beyond the reports.** The model beats the best report-derived label source on effusion, fracture, medial OA and lateral meniscus.
  - It trails on synovitis and PF OA. Radiologists reading the images flag these findings far more often than reports mention them.
- **Gold tracks the hidden test set:** 0.907 on gold vs 0.906 on the private leaderboard, even with only 58 gold studies.

## Repository structure

```
knee-mri-abnormality-detection/
├── configs/                 # YAML: shared training settings + one file per model family
├── src/kneemri/
│   ├── preprocessing.py     # DICOM -> canonical, cropped, normalised volumes
│   ├── cache.py             # builds the uint8 training cache (ZIP shards)
│   ├── labels/              # keyword labeler, LLM grader, calibration + blending
│   ├── data.py              # dataset, folds, GPU augmentation, anatomical mirror
│   ├── model.py             # KneeNet
│   ├── train.py             # one fold of one family
│   ├── inference.py         # offline TTA + rank ensemble + validated submission
│   └── metrics.py
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_report_labelling.ipynb
│   ├── 03_preprocess_and_train_kaggle.ipynb
│   ├── 04_inference_kaggle.ipynb
│   └── 05_results.ipynb
├── reports/                 # figures + metric tables
├── docs/pipeline.png
└── tests/                   # unit tests (labeler, model, augmentation, ensemble, calibration)
```

## Quick start

```bash
git clone https://github.com/lucky07-07/knee-mri-abnormality-detection.git
cd knee-mri-abnormality-detection
pip install -r requirements.txt && pip install -e .
pytest -q
```

**Data:** the data is from the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) competition, and its rules do not allow redistribution. Download it with the Kaggle API and see [data/README.md](data/README.md).

**Reproduce:**
1. Notebooks 01 and 02 run locally on `train.csv`.
2. Notebooks 03 and 04 run on Kaggle with GPU (2× T4) and hold the DICOM-heavy steps: cache, LLM labels, training, inference.
3. Notebook 05 rebuilds every result above from the saved predictions.

Train one fold:

```bash
python -m kneemri.train --config configs/native_mirror.yaml --fold 0 \
    --train-csv data/train.csv --labels outputs/labels/blended_labels.csv \
    --cache <cache_part_0> <cache_part_1> <cache_part_2>
```

## Tech stack

Python · PyTorch · timm (ConvNeXt) · Hugging Face Transformers + bitsandbytes (Qwen2.5-14B, 4-bit) · pydicom · OpenCV · scikit-learn · pandas · Matplotlib / Seaborn · Kaggle GPU (2× T4)

## Author

**Anil Kumar** · [GitHub](https://github.com/lucky07-07)

## License

The code is released under the MIT License. The competition data is not included and is subject to the competition rules.
