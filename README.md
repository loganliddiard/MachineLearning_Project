# How to Install and Run

This repo contains one main notebook:
- `RoadCondition_Experiments.ipynb` (road-condition dataset baselines + attention placement grid)

## 0) IMPORTANT NOTE
- Due to differences in hardware and GPU's the train/val/test accuracies can vary between devices. 
- We talked with the professor about this, and he said to just leave a note explaining it.
- We ran ours on a linux system with a GTX 3080.

## 1) Clone
```bash
git clone https://github.com/loganliddiard/MachineLearning_Project.git
cd MachineLearning_Project
```

## 2) Create a Python virtual environment (venv)

### Windows (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Windows (Git Bash)
```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
```

## 3) Install Python dependencies
```bash
pip install -r requirements.txt
```

## 4) Road Condition Experiments

### Dataset folder (required)
Put the dataset folder at the repo root named **exactly**:
`UDOT WINTER ROAD CONDITIONS.v1i.folder`

### Unzip / extract the dataset correctly (important)
1) Download the dataset as a `.zip`.
2) Extract it so the folder sits **directly** under the repo root.
3) After extracting, confirm you have this shape (no extra nesting):

```text
MachineLearning_Project/
  UDOT WINTER ROAD CONDITIONS.v1i.folder/
    dataset_index.csv
    train/
    valid/
    test/
```

Common unzip issue: you end up with a double-nested folder like:
`UDOT WINTER ROAD CONDITIONS.v1i.folder/UDOT WINTER ROAD CONDITIONS.v1i.folder/train/...`

If that happens, move the *inner* `UDOT WINTER ROAD CONDITIONS.v1i.folder` contents up one level so `train/`, `valid/`, `test/`, and `dataset_index.csv` are directly inside the first folder.

### Dataset layout (required)
Expected structure:
- `UDOT WINTER ROAD CONDITIONS.v1i.folder/train/<label>/<image_id>`
- `UDOT WINTER ROAD CONDITIONS.v1i.folder/valid/<label>/<image_id>`
- `UDOT WINTER ROAD CONDITIONS.v1i.folder/test/<label>/<image_id>`

CSV index file (required by the notebook):
- `UDOT WINTER ROAD CONDITIONS.v1i.folder/dataset_index.csv` with columns `image_id, split, label`

Note: the repo also has a top-level `dataset_index.csv`, but the notebook code expects the one inside the dataset folder.

### (Optional) Preprocess the dataset
Resize images to 224×224 (recommended):
```bash
python preprocess.py
```

### (Optional) Rebuild `dataset_index.csv`
If you need to generate/rebuild `UDOT WINTER ROAD CONDITIONS.v1i.folder/dataset_index.csv` from the folder structure:
```bash
python dummy_model.py
```

### Run the notebook
Launch Jupyter and open the road-condition experiment notebook:
```bash
jupyter notebook RoadCondition_Experiments.ipynb
```

In VS Code: open `RoadCondition_Experiments.ipynb`, select the `.venv` kernel, then **Run All**.

### Notes
- The default notebook section trains the three CNN baselines (LeNet224, SmallAlexNet224, ResNet18).
- The attention placement grid runs additional training runs and can take a while; reduce `fit_cfg.epochs` first if you just want a quick sanity check.