
# In-Context Learning for Deepfake Detection

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
