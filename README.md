# How to Install and Run

## 1. Clone the Repository
```bash
git clone https://github.com/Toran625/CS_5640_Final_Project.git
cd CS_5640_Final_Project
```

## 2. Create and Activate a Virtual Environment
### Using conda (what I used, but can be done using any other virtual environment)
```bash
conda create -n attention_env python=3.13
conda activate attention_env
```

## 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 4. Run the Project
### Option 1: Launch Jupyter Notebook
```bash
jupyter notebook AttentionPlacement_CNN.ipynb
```
### Option 2: Run In VSCode Environment
Open in VSCode using the Jupyter Notebook extension and RunAll

## Road Condition Experiments (New)

### Dataset layout (same as scripts)
- Put the dataset folder at the repo root named: `UDOT WINTER ROAD CONDITIONS.v1i.folder`
- Expected structure:
	- `UDOT WINTER ROAD CONDITIONS.v1i.folder/train/<label>/<image_id>`
	- `UDOT WINTER ROAD CONDITIONS.v1i.folder/valid/<label>/<image_id>`
	- `UDOT WINTER ROAD CONDITIONS.v1i.folder/test/<label>/<image_id>`
- CSV index file:
	- `UDOT WINTER ROAD CONDITIONS.v1i.folder/dataset_index.csv` (columns: `image_id, split, label`)

### Recommended run order
1. (Optional) Resize images to 224×224 using `preprocess.py`.
2. (If needed) Rebuild the CSV index using `dummy_model.py` (it generates `dataset_index.csv`).
3. Open and run `RoadCondition_Experiments.ipynb`.

### Notes
- The default notebook run trains only basic CNN baselines.
- The attention placement grid-search code is included but disabled by default.