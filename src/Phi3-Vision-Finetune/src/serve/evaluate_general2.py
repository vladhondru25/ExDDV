import json
import os
import requests 
import sys
from PIL import Image 

import cv2
import numpy as np
import pandas as pd
import torch
from evaluate import load
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, BertTokenizer, BertModel
from scipy import spatial
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

sys.path.append("/home/eivor/biodeep/xAI_deepfake/src/Phi3-Vision-Finetune/src")
from training.data import ExplainableDataset, find_closest_values


querry_prompt = "Analyze the face in the image. Identify any deepfake artifacts (if any), focusing specifically on the affected parts of the face mentioned. Provide a short and direct explanation highlighting the inconsistencies or manipulations."

st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')


model_id = "microsoft/Phi-3.5-vision-instruct" 
# Note: set _attn_implementation='eager' if you don't have flash_attn installed
model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  # "/mnt/home/fmi2/vladh/phi3v-xai/output/model_merged", 
  # "/home/eivor/biodeep/xAI_deepfake/src/Phi3-Vision-Finetune/output/lora_vision_test", 
  device_map="cuda", 
  trust_remote_code=True, 
  torch_dtype="auto", 
  _attn_implementation='flash_attention_2'    
)

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
processor = AutoProcessor.from_pretrained(model_id,
  trust_remote_code=True, 
  num_crops=4
)



def _obtain_path(row):
  FACEFORENSINCS_PATH = "/home/eivor/data/dp/manipulated_videos"
  DEEPFAKECHALLENGE_PATH = "/home/eivor/data/deepfake_dataset_challenge"
  DEPERFORENSINCS_PATH = "/home/eivor/data/ffpp/manipulated_sequences"
  BIODEEPAV_PATH = "/home/eivor/data/BioDeepAV/fake/videos"
  REAL_VIDEOS_PATH = "/home/eivor/data/dp/original_data/original/data/original_sequences/youtube/c23/videos"
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
csv_data_fake = pd.read_csv("/home/eivor/biodeep/xAI_deepfake/dataset5.csv")
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
  if idx == 5:
    break
  images = []
  placeholder = ""
  movie_name = video_metadata["movie_name"]
  
  cap = cv2.VideoCapture(movie_name)
  movie_no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  frames_indices = np.linspace(start=0, stop=movie_no_frames-1, num=5)
  frames_indices = frames_indices.astype(int)
  
  click_locations = video_metadata["click_locations"]
  click_locations = json.loads(click_locations)
  click_locations_time = [int(t) for t in list(click_locations.keys())]
  closest_frame_indices = find_closest_values(frames_indices, click_locations_time)

  # Note: if OOM, you might consider reduce number of frames in this example.
  keypoints = []
  for i in range(5):
      # url = f"https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg" 
      # images.append(Image.open(requests.get(url, stream=True).raw))

      # Set the current frame position to the desired frame
      frame_ind = frames_indices[i]
      cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)

      # Read the frame
      ret, frame = cap.read()

      # frame = cv2.resize(frame, (WIDTH, HEIGHT))
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      keypoint = click_locations[str(closest_frame_indices[i])]
      
      if ExplainableDataset.flag_draw_keypoints:
        frame = ExplainableDataset.draw_keypoints(frame, keypoint)
        
      keypoints.append(keypoint)
      
      frame = Image.fromarray(frame)

      images.append(frame)
      placeholder += f"<|image_{i+1}|>\n"
      
  keypoints_kwargs = {}
  if ExplainableDataset.flag_use_masking:
    # keypoints_kwargs = {"keypoints": keypoints, "keypoints_fn": ExplainableDataset.apply_mask, "use_hard_mask": True, "radius": ExplainableDataset.RADIUS}
    keypoints_kwargs = {"keypoints": keypoints, "keypoints_fn": ExplainableDataset.apply_mask, "use_hard_mask": False, "radius": ExplainableDataset.RADIUS}
    # keypoints_kwargs = {"keypoints": keypoints, "keypoints_fn": ExplainableDataset.draw_keypoints, "radius": 17}

  messages = [
      # {"role": "user", "content": placeholder+ExplainableDataset.video_question},
      {"role": "user", "content": placeholder+querry_prompt},
  ]

  prompt = processor.tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
  )

  inputs = processor(prompt, images, return_tensors="pt", **keypoints_kwargs).to("cuda:0") 

  generation_args = { 
    "max_new_tokens": 1000, 
    "temperature": 0.0, 
    "do_sample": False, 
  } 

  generate_ids = model.generate(**inputs, 
    eos_token_id=processor.tokenizer.eos_token_id, 
    **generation_args
  )

  # remove input tokens 
  generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
  response = processor.batch_decode(generate_ids, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False)[0]
  
  gt_sentences.append(video_metadata["text"])
  pred_sentences.append(response.strip())
  
  # if idx == 5:
  #   break
  
# Write down predictions and gt
json_data = []
for gt_sentence, pred_sentence in zip(gt_sentences,pred_sentences):
  json_data.append({"gt": gt_sentence, "pred": pred_sentence})
with open("/home/eivor/biodeep/xAI_deepfake/src/Phi3-Vision-Finetune/output/predictions_baseline.json", "w", encoding='utf-8') as out_file:
  json.dump(json_data, out_file, ensure_ascii=False, indent=4)
  
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
