# AI Paper Similarity Search
AI Paper Similarity Search allows users to search for academic papers similar to their query. The application uses machine learning to rank papers based on their relevance and provides direct links to the papers.

**Live Demo**: [AI Paper Similarity Search App](https://napronald.github.io/AI-Paper-Similarity-Search-App/)

**Note:** The web app loads small subsets of data to avoid overwhelming the browser. To demo the full model containing all [2 million embeddings](https://drive.google.com/drive/u/2/folders/1iuOpyaHjqTYuCuKdDz4uOY_fJPvN3GbT) you can follow the installation instructions below and run the `eval.py` script in the `training` folder.

## Web App Video Demo
https://github.com/user-attachments/assets/49507189-9110-426b-9adb-9773761f8727

## Full Model Demo

https://github.com/user-attachments/assets/1e9c1328-0629-464b-b24e-b0743b8be02e

## Installation

First, ensure you have git and conda installed:

```sh
# Step 1: Create and activate Conda environment
conda create -n AIPSS python=3.8
conda activate AIPSS

# Step 2: Clone the repository
git clone https://github.com/napronald/AI-Paper-Similarity-Search-App.git
cd AI-Paper-Similarity-Search-App/training

# Step 3: Install dependencies
pip install -r requirements.txt
```

Then, you can either run `main.py` to generate the metadata yourself or download the precomputed data using:

```sh
# Step 3: Download metadata
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1iuOpyaHjqTYuCuKdDz4uOY_fJPvN3GbT
mv AI-Paper-Similarity-Search/* .
rm -r AI-Paper-Similarity-Search
```

This will download all the files needed into the current directory to run `eval.py`.

```sh
# Step 4: Add cosmetic effects
pip install textwrap3
pip install colorama
python eval.py
```

# Support
If you found this project interesting and helpful, please consider giving it a star 🌟 to support its development. 
