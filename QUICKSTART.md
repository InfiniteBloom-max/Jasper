# JASPER: Quick Start Guide

Get the textile analysis running in 5 minutes.

## Step 1: Download & Extract

```bash
# Download this repository
unzip jasper_reproducible.zip
cd jasper_reproducible
```

## Step 2: Setup Environment

### Option A: Automated Setup (Linux/Mac)

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### Option B: Manual Setup (All Platforms)

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Get Dataset

### Using Kaggle API (Recommended)

```bash
# Install Kaggle
pip install kaggle

# Setup Kaggle credentials
# 1. Go to https://www.kaggle.com/settings
# 2. Click "Create New API Token"
# 3. Save kaggle.json to ~/.kaggle/ (Linux/Mac) or %USERPROFILE%\.kaggle\ (Windows)

# Download dataset
kaggle datasets download -d ronithrr/jasper-japanese-x-sri-lankan-textile-image-dataset
unzip jasper-japanese-x-sri-lankan-textile-image-dataset.zip -d data/
```

### Manual Download

1. Visit: https://www.kaggle.com/datasets/ronithrr/jasper-japanese-x-sri-lankan-textile-image-dataset
2. Click "Download"
3. Extract to `data/` folder

## Step 4: Run Analysis

### Quick Test (100 images, ~2 minutes)

```bash
python main_analysis.py --dataset data/ --output results_test --sample 100 --verbose
```

### Full Analysis (2000 images, ~15-30 minutes)

```bash
python main_analysis.py --dataset data/ --output results --verbose
```

## Step 5: View Results

```bash
# Statistical report
cat results/statistical_report.txt

# Open figures
open results/figures/  # Mac
start results/figures/  # Windows
xdg-open results/figures/  # Linux
```

## Interactive Analysis

```bash
jupyter notebook notebooks/interactive_analysis.ipynb
```

---

## Troubleshooting

### "No module named cv2"

```bash
pip install opencv-python
```

### "No PNG files found"

Check dataset path. Should be:
```
data/
  japanese_textiles/
    *.png
  sri_lankan_textiles/
    *.png
```

### Slow performance

Use `--sample` for testing:
```bash
python main_analysis.py --dataset data/ --sample 50
```

---

## What You Get

After running, you'll have:

- **extracted_features.csv**: All 44 features for every image
- **statistical_comparison.csv**: T-tests, p-values, Cohen's d
- **key_differentiators.csv**: Features with very large effect sizes
- **statistical_report.txt**: Human-readable summary
- **figures/**: All publication-quality visualizations

## Key Findings to Verify

From the paper, you should see:

- **warm_cool_score**: Japanese (-0.111) vs Sri Lankan (0.178), d ≈ 3.059
- **texture_complexity**: Japanese > Sri Lankan, d ≈ 1.590
- **symmetry_score**: Japanese (0.055) < Sri Lankan (0.172)

---

For detailed documentation, see README.md
