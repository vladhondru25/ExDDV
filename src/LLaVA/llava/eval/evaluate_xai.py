import json
import os
import requests 
import sys
from PIL import Image 

import cv2
import numpy as np
import pandas as pd
import torch

import nltk
from evaluate import load
from tqdm import tqdm
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
from scipy import spatial
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.train.train import XAIDataset


# sys.path.append("/home/eivor/biodeep/xAI_deepfake/src/Phi3-Vision-Finetune/src")
# from training.data import ExplainableDataset, find_closest_values

# model_path = "/home/eivor/biodeep/xAI_deepfake/src/LLaVA/checkpoints/merged_lora"

EXP_NAME = "llava-v1.5-7b-task-lora-new-baseline"
model_path = f"/mnt/home/fmi2/vladh/llava_xai/checkpoints/{EXP_NAME}"
model_base = "liuhaotian/llava-v1.5-7b" 
model_name = get_model_name_from_path(model_path)

conv_mode = "llava_v1"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path,
    model_base,
    model_name,
    device_map='cuda'
)

# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path, model_base, model_name,
#     use_flash_attn=True,
#     device_map="cuda"
# )

st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')


def find_closest_values(list1, list2):
    closest_values = []
    for num in list1:
        closest = min(list2, key=lambda x: abs(x - num))  # Find the closest value in list2
        closest_values.append(closest)
    return closest_values

def _obtain_path(row):
    FACEFORENSINCS_PATH = "/mnt/home/fmi2/vladh/data/dp/manipulated_videos"
    DEEPFAKECHALLENGE_PATH = "/mnt/home/fmi2/vladh/data/deepfake_dataset_challenge"
    DEPERFORENSINCS_PATH = "/mnt/home/fmi2/vladh/data/ffpp/manipulated_sequences"
    BIODEEPAV_PATH = "/mnt/home/fmi2/vladh/data/BioDeepAV/fake/videos"
    REAL_VIDEOS_PATH = "/mnt/home/fmi2/vladh/data/original_sequences/youtube/c23/videos"
    if row["dataset"] == "Farceforensics++":
        return os.path.join(FACEFORENSINCS_PATH, row["manipulation"], row["movie_name"])
    elif row["dataset"] == "Deeperforensics":
        return os.path.join(DEPERFORENSINCS_PATH, row["manipulation"], "c23/videos", row["movie_name"])
    elif row["dataset"] == "DeepfakeDetection":
        return os.path.join(DEEPFAKECHALLENGE_PATH, row["manipulation"], row["movie_name"])
    elif row["dataset"] == "BioDeepAV":
        return os.path.join(BIODEEPAV_PATH, row["movie_name"])
    elif row["dataset"] == "Original":
        return os.path.join(REAL_VIDEOS_PATH, row["movie_name"])
        
csv_data_fake = pd.read_csv("/mnt/home/fmi2/vladh/data/dataset6.csv")
csv_data_fake["movie_name"] = csv_data_fake.apply(_obtain_path, axis=1)
csv_data_fake = csv_data_fake.sample(frac=1, random_state=0).reset_index()

split = "test"
if split == "train":
    csv_data_fake = csv_data_fake[csv_data_fake["split"] == "train"]
elif split == "val":
    csv_data_fake = csv_data_fake[csv_data_fake["split"] == "val"]
elif split == "test":
    csv_data_fake = csv_data_fake[csv_data_fake["split"] == "test"]

pred_sentences = []
gt_sentences = []
for idx, video_metadata in tqdm(csv_data_fake.iterrows(), desc="Generating the predictions", total=len(csv_data_fake)):
    images = []
    placeholder = ""
    movie_name = video_metadata["movie_name"]

    cap = cv2.VideoCapture(movie_name)
    movie_no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # frames_indices = np.linspace(start=0, stop=movie_no_frames-1, num=5)
    # frames_indices = frames_indices.astype(int)
    frames_indices = [movie_no_frames//2]

    click_locations = video_metadata["click_locations"]
    click_locations = json.loads(click_locations)
    click_locations_time = [int(t) for t in list(click_locations.keys())]
    
    closest_frame_indices = find_closest_values(frames_indices, click_locations_time)
    keypoint = [click_locations[str(c)] for c in closest_frame_indices][0]

    cap.set(cv2.CAP_PROP_POS_FRAMES, movie_no_frames//2)
    # Read the frame
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    
    qs = "What is wrong in the image?"
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if model.config.mm_use_im_start_end:
        qs = image_token_se + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images = [frame]
    image_sizes = [x.size for x in images]
    
    keypoint_args = {}
    if XAIDataset.use_keypoints:
        keypoint_args["keypoints_fn"] = XAIDataset.apply_mask
        keypoint_args["keypoints_hard_masking"] = XAIDataset.use_hard_masking
        keypoint_args["keypoint"] = keypoint
    
    images_tensor = process_images(
        images,
        image_processor,
        model.config,
        **keypoint_args
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0.0,
            top_p=None,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True,
        )

    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
  
    pred_sentences.append(response)
    gt_sentences.append(video_metadata["text"])

  
# Write down predictions and gt
json_data = []
for gt_sentence, pred_sentence in zip(gt_sentences,pred_sentences):
  json_data.append({"gt": gt_sentence, "pred": pred_sentence})
with open(f"/mnt/home/fmi2/vladh/llava_xai/checkpoints/{EXP_NAME}/predictions.json", "w", encoding='utf-8') as out_file:
  json.dump(json_data, out_file, ensure_ascii=False, indent=4)
  
# Sentence transformer
ground_truth_embds = []
predicted_embds = []
with torch.no_grad():
    for idx,d in tqdm(enumerate(pred_sentences), desc=f"Computing the pred embeddings", total=len(pred_sentences)):
        embd = st_model.encode(d, show_progress_bar=False, convert_to_tensor=False)
        predicted_embds.append(embd)
  
    for idx,d in tqdm(enumerate(gt_sentences), desc=f"Computing the gt embeddings", total=len(gt_sentences)):
        embd = st_model.encode(d, show_progress_bar=False, convert_to_tensor=False)
        ground_truth_embds.append(embd)

def cosine_similarity(y_trues, y_preds):
    return np.mean([
        1 - spatial.distance.cosine(y_true, y_pred) 
        for y_true, y_pred in zip(y_trues, y_preds)
    ])

print(f"Sentence embedding cosine similarity: ", cosine_similarity(ground_truth_embds, predicted_embds))

# BERT Score
bertscore = load("bertscore")
scores = bertscore.compute(predictions=pred_sentences, references=gt_sentences, lang="en", rescale_with_baseline=True, idf=True, model_type="distilbert-base-uncased")
print(f"BERT score f1: ", np.mean(scores["f1"])) # 0.32632499808445575


# Compute BLEU scores for each pair
smooth = SmoothingFunction().method1
bleu_scores = [
    sentence_bleu([gt.split()], pred.split(), smoothing_function=smooth, weights=[0.5, 0.5])
    for gt, pred in zip(gt_sentences, pred_sentences)
]
average_bleu = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU Score: {average_bleu:.4f}")

# Compute METEOR score
# meteor_scores = [
#     meteor_score([gt.split()], pred.split())
#     for gt, pred in zip(gt_sentences, pred_sentences)
# ]
# average_meteor = sum(meteor_scores) / len(meteor_scores)
# print(f"Average METEOR Score: {average_meteor:.4f}")
meteor = load('meteor')
meteor_scores = [
    meteor.compute(predictions=[pred], references=[gt])["meteor"]
    for gt, pred in zip(gt_sentences, pred_sentences)
]
average_meteor = sum(meteor_scores) / len(meteor_scores)
print(f"Average METEOR Score: {average_meteor:.4f}")

# Compute ROUGE score
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = [
    scorer.score(gt, pred)
    for gt, pred in zip(gt_sentences, pred_sentences)
]

average_rouge_1 = np.mean([r['rouge1'].fmeasure for r in rouge_scores])
average_rouge_2 = np.mean([r['rouge2'].fmeasure for r in rouge_scores])
average_rouge_3 = np.mean([r['rougeL'].fmeasure for r in rouge_scores])

print(f"ROUGE-1: {average_rouge_1:.4f}")
print(f"ROUGE-2: {average_rouge_2:.4f}")
print(f"ROUGE-L: {average_rouge_3:.4f}")
