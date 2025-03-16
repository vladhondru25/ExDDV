
# ExDDV: A New Dataset for Explainable Deepfake Detection in Video
This repository contains the necessary code to run the experiments in the paper "ExDDV: A New Dataset for Explainable Deepfake Detection in Video".

## Dataset
The csv file containing the annotations is provided to anyone for research purposes. Feel free to contact us so access can be provided.

## Fine-tuning for Deepfake Detection
There are three different folders inside the `src` folder containing the three models we used: `LAVIS` (BLIP-2), `LLaVA` and `Phi3-Vision-Finetune`. These are forked from the following repositories: [LAVIS](https://github.com/salesforce/LAVIS), [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main) and [Phi3-Vision-Finetune](https://github.com/2U1/Phi3-Vision-Finetune).

### How to run
Each model is trained as per its corresponding repository. We used the respective training scripts: [BLIP-2](src/LAVIS/lavis/projects/blip2/train/pretrain_stage2_xai.yaml), [LLaVA](src/LLaVA/scripts/v1_5/finetune_task_lora.sh) and [Phi3-Vision](src/Phi3-Vision-Finetune/scripts/finetune_lora_vision.sh). The only modification needed is to replaced the paths to the datasets in the corresponding files.

!Note for Phi3-Vision: The file `processing_phi3_v.py` from HuggingFace Transformers must be replaced with the script from [here](src/Phi3-Vision-Finetune/proccessing_phi3_v.py).

## In-Context Learning for Deepfake Detection

The current implementation uses LLaVA 1.5, BLIP-2 and PHI3-Vision. 

![image](https://github.com/user-attachments/assets/9ea67b6c-9412-4de3-9e65-2886276ee44f)



**Note:**  
The code for each model is provided in separate Jupyter notebooks:
- LLAVA: `llava_incontext.ipynb`
- PHI3-Vision: `phi_incontext.ipynb`
- BLIP-2: `blip_incontext.ipynb`

## Overview

- **Extracting visual embeddings:** from training and test video frames using a pre-trained vision model.
- **Constructing in-context prompts:** by retrieving the top-k most similar training annotations.
- **Analyzing deepfake artifacts:** in test frames using a language-vision model (e.g., LLAVA).
- **Optionally applying spatial masks:** to focus the analysis on specific facial regions.

## Pipeline Components

### Data Preparation and Normalization
- Dataset names (e.g., "FaceForensics++").
- Locates movie files based on dataset and manipulation type.
- Extracts training and test frames from a CSV file.

### Embedding Extraction
- Uses a vision model (e.g., CLIP with RN101) with support for different extraction layers ("first", "middle", "last") to compute image embeddings.

### In-Context Prompt Construction
- Generates custom prompts from training annotations using multiple prompt templates (different version will influence the level of detail in the response).

### Deepfake Analysis
- Loads the model and evaluates test frames using generated prompts.
- Applies optional hard or soft masks around keypoints.

### Evaluation and Results
- Compares test frame embeddings with training embeddings using cosine similarity.
- Retrieves top-k training examples to form a contextual prompt.
- Saves detailed results to CSV files.

## Getting Started

### Requirements

*Requirements will differ based on the vision model used. We have followed the indications from the original paper. For LLAVA, for example, use the following steps:*

1. Clone the LLAVA repository and navigate to the LLaVA folder:
   ```bash
   git clone https://github.com/haotian-liu/LLaVA.git
   cd LLaVA
   ```
2. Install the package in a Conda environment:
   ```bash
   conda create -n llava python=3.10 -y
   conda activate llava
   pip install --upgrade pip  # enable PEP 660 support
   pip install -e .
   ```

For other vision models (e.g., BLIP-2, PHI3-Vision), refer to their official installation instructions.

### Data Preparation

1. **Dataset CSV:**  
   Ensure your dataset CSV (e.g., `dataset_last.csv`) includes columns such as `movie_name`, `dataset`, `manipulation`, `movie_path`, `click_locations`, and `text`. The CSV should be split into training, validation(not used here), and test sets.

2. **Video Files:**  
   Place your video files in the `data/` folder. The directory structure should follow the dataset names (e.g., `data/Faceforensics++/`).


## Results and Evaluation

The pipeline outputs CSV files containing:
- Test video information (file path and frame number).
- Ground truth annotations.
- Contextual prompts from top-k training annotations.
- Cosine similarity scores.
- Deepfake analysis results.


```
