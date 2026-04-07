# How to Install and Run

This repo contains one main notebook:
- `RoadCondition_Experiments.ipynb` (road-condition dataset baselines + attention placement grid)

## 1) Clone
```bash
git clone https://github.com/loganliddiard/MachineLearning_Project.git
cd MachineLearning_Project
```

## 2) Create a Python virtual environment (venv)
Use a standard Python environment.

### Windows
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

### Dataset layout (required)
Put the dataset folder at the repo root named exactly:
`UDOT WINTER ROAD CONDITIONS.v1i.folder`

Expected structure:
- `UDOT WINTER ROAD CONDITIONS.v1i.folder/train/<label>/<image_id>`
- `UDOT WINTER ROAD CONDITIONS.v1i.folder/valid/<label>/<image_id>`
- `UDOT WINTER ROAD CONDITIONS.v1i.folder/test/<label>/<image_id>`

CSV index file (required by the notebook):
- `UDOT WINTER ROAD CONDITIONS.v1i.folder/dataset_index.csv` with columns `image_id, split, label`

### (Optional) Preprocess the dataset
Resize images to 224×224 (recommended):
```bash
python preprocess.py
```

If you need to generate/rebuild `dataset_index.csv` from the folder structure:
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