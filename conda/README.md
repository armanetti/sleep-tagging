# Prepare the Environment

Suggested Python version: **≥ 3.11**

This project provides two ways to configure the environment: using **conda** (recommended) or **pip**.

---

## Using `conda` (recommended)

### 1. Create the environment

```bash
conda env create -f <PATH_TO_REPO>/conda/environment.yml
```

### 2. Activate the environment

```bash
conda activate sleep-tagging-env
```

### (Optional) Create an alias (POSIX/UNIX)

```bash
echo "alias sleep-tagging-env='conda activate sleep-tagging-env'" >> ~/.bashrc && source ~/.bashrc
```

### (Optional) Install the environment as a Jupyter kernel

```bash
conda activate sleep-tagging-env && python -m ipykernel install --user --name SleepTagging --display-name "SleepTagging"
```

### (Optional) Install Jupyter inside this environment

```bash
pip install jupyter
```

**Jupyter documentation:** [https://docs.jupyter.org/en/stable/install/notebook-classic.html](https://docs.jupyter.org/en/stable/install/notebook-classic.html)

**Note:** Use `pip3` in environments where multiple Python versions are installed.

---

## Using `pip`

### Install dependencies

```bash
pip install -r <PATH_TO_REPO>/conda/requirements.txt
```

### (Optional) Add the kernel to Jupyter

```bash
python -m ipykernel install --user --name SleepTagging --display-name "SleepTagging"
```

---

## Verify Installation

You can confirm that dependencies were installed correctly:

```bash
python -c "import numpy, pandas; print('Environment OK')"
```

Or launch Jupyter Notebook:

```bash
jupyter notebook
```

---

## How to Remove This Environment

### 1. If active, deactivate:

```bash
conda deactivate
```

### 2. Remove the conda environment:

```bash
conda remove -n sleep-tagging-env --all -y
```

---

#### Created for Brain Hack 2025 IMT Lucca