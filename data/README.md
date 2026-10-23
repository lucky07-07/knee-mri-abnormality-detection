# Data

The data comes from the Kaggle competition **[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)**.
Its rules do not allow redistribution, so nothing here is committed.

```bash
kaggle competitions download -c rsna-knee-abnormality-detection -f train.csv -p data/
```

Expected layout:

```
data/
├── train.csv          # StudyInstanceUID, Report, 12 finding columns (filled only for the 58 gold studies)
└── labels/            # saved outputs of the GPU label passes (see notebook 02)
    ├── 09_report_llm_grades_v2.csv     # Qwen2.5-14B hard + expected grades
    ├── llm_rubric_report_labels.csv    # rubric LLM pass, soft labels
    └── 10_report_labels.csv            # teacher (image-model) soft labels
```

The DICOM images (`train_series/`, `test_series/`) are only used on Kaggle (notebooks 03 and 04).
