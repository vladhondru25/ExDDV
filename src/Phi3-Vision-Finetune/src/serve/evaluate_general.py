import json
import os
import requests 
import sys
from PIL import Image 

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, BertTokenizer, BertModel
from scipy import spatial
from sentence_transformers import SentenceTransformer

sys.path.append("/home/eivor/biodeep/xAI_deepfake/src/Phi3-Vision-Finetune/src")
from training.data import ExplainableDataset, find_closest_values


USE_SENTENCE_EMBEDDER = True

if USE_SENTENCE_EMBEDDER:
  st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
else:
  bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  bert_model = BertModel.from_pretrained("bert-base-uncased")

model_id = "microsoft/Phi-3.5-vision-instruct" 

# Note: set _attn_implementation='eager' if you don't have flash_attn installed
model = AutoModelForCausalLM.from_pretrained(
  # model_id, 
  "/home/eivor/biodeep/xAI_deepfake/src/Phi3-Vision-Finetune/output/model_merged", 
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
    if row["dataset"] == "Farceforensics++":
        return os.path.join(FACEFORENSINCS_PATH, row["manipulation"], row["movie_name"])
    elif row["dataset"] == "Deeperforensics":
        return os.path.join(DEPERFORENSINCS_PATH, row["manipulation"], "c23/videos", row["movie_name"])
    elif row["dataset"] == "DeepfakeDetection":
        return os.path.join(DEEPFAKECHALLENGE_PATH, row["manipulation"], row["movie_name"])
csv_data_fake = pd.read_csv("/home/eivor/biodeep/xAI_deepfake/dataset2.csv")
csv_data_fake["movie_name"] = csv_data_fake.apply(_obtain_path, axis=1)
csv_data_fake = csv_data_fake.sample(frac=1, random_state=0).reset_index()

# csv_data_fake = csv_data_fake[100:] # if split == "train":
csv_data_fake = csv_data_fake[:100] # elif split == "val":

pred_sentences = []
gt_sentences = []
for idx, video_metadata in tqdm(csv_data_fake.iterrows(), desc="Generating the predictions", total=len(csv_data_fake)):
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
    keypoints_kwargs = {"keypoints": keypoints, "keypoints_fn": ExplainableDataset.apply_mask, "use_hard_mask": True, "radius": ExplainableDataset.RADIUS}
    # keypoints_kwargs = {"keypoints": keypoints, "keypoints_fn": ExplainableDataset.draw_keypoints, "radius": 17}

  messages = [
      {"role": "user", "content": placeholder+ExplainableDataset.video_question},
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
with open("/home/eivor/biodeep/xAI_deepfake/src/Phi3-Vision-Finetune/output/lora_vision_test/predictions.json", "w", encoding='utf-8') as out_file:
  json.dump(json_data, out_file, ensure_ascii=False, indent=4)
  
ground_truth_embds = []
predicted_embds = []

if USE_SENTENCE_EMBEDDER:
  with torch.no_grad():
    for idx,d in tqdm(enumerate(pred_sentences), desc=f"Computing the pred embeddings", total=len(pred_sentences)):
        embd = st_model.encode(d, show_progress_bar=False, convert_to_tensor=False)
        predicted_embds.append(embd)
  
    for idx,d in tqdm(enumerate(gt_sentences), desc=f"Computing the gt embeddings", total=len(gt_sentences)):
        embd = st_model.encode(d, show_progress_bar=False, convert_to_tensor=False)
        ground_truth_embds.append(embd)
else:
  with torch.no_grad():
    for idx,d in tqdm(enumerate(pred_sentences), desc=f"Computing the pred embeddings", total=len(pred_sentences)):
      encoded_input = bert_tokenizer(d, return_tensors='pt')
      output = bert_model(**encoded_input)
      predicted_embds.append(output.pooler_output.cpu().squeeze().numpy())
    
    for idx,d in tqdm(enumerate(gt_sentences), desc=f"Computing the gt embeddings", total=len(gt_sentences)):
      encoded_input = bert_tokenizer(d, return_tensors='pt')
      output = bert_model(**encoded_input)
      ground_truth_embds.append(output.pooler_output.cpu().squeeze().numpy())

def cosine_similarity(y_trues, y_preds):
    return np.mean([
        1 - spatial.distance.cosine(y_true, y_pred) 
        for y_true, y_pred in zip(y_trues, y_preds)
    ])

print(cosine_similarity(ground_truth_embds, predicted_embds))
