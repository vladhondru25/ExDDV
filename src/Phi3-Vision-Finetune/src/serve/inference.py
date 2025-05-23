import os
import requests 
from PIL import Image 

import cv2
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 

model_id = "microsoft/Phi-3.5-vision-instruct" 

# Note: set _attn_implementation='eager' if you don't have flash_attn installed
model = AutoModelForCausalLM.from_pretrained(
#   model_id, 
  "/home/eivor/biodeep/xAI_deepfake/src/Phi3-Vision-Finetune/output/model_merged", 
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

images = []
placeholder = ""

def _obtain_path(row):
  FACEFORENSINCS_PATH = "/home/eivor/data/dp/manipulated_videos"
  DEEPFAKECHALLENGE_PATH = "/home/eivor/data/deepfake_dataset_challenge"
  if row["dataset"] == "Farceforensics++":
      return os.path.join(FACEFORENSINCS_PATH, row["manipulation"], row["movie_name"])
  elif row["dataset"] == "DeepfakeDetection":
      return os.path.join(DEEPFAKECHALLENGE_PATH, row["manipulation"], row["movie_name"])
csv_data_fake = pd.read_csv("/home/eivor/biodeep/xAI_deepfake/dataset.csv")
csv_data_fake["movie_name"] = csv_data_fake.apply(_obtain_path, axis=1)
video_metadata = csv_data_fake.loc[0]
movie_name = video_metadata["movie_name"]
cap = cv2.VideoCapture(movie_name)
movie_no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frames_indices = np.linspace(start=0, stop=movie_no_frames-1, num=5)
frames_indices = frames_indices.astype(int)

# Note: if OOM, you might consider reduce number of frames in this example.
for i in range(1, 5):
    # url = f"https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg" 
    # images.append(Image.open(requests.get(url, stream=True).raw))

    # Set the current frame position to the desired frame
    frame_ind = frames_indices[i]
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)

    # Read the frame
    ret, frame = cap.read()

    # frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    images.append(frame)
    placeholder += f"<|image_{i}|>\n"

messages = [
    # {"role": "user", "content": placeholder+"Determine why the deck of slides are from deepfake video."},
    {"role": "user", "content": placeholder+"What is wrong in this video?"},
    # {"role": "user", "content": placeholder+"Summarize the deck of slides."},
]

prompt = processor.tokenizer.apply_chat_template(
  messages, 
  tokenize=False, 
  add_generation_prompt=True
)

inputs = processor(prompt, images, return_tensors="pt").to("cuda:0") 

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

print(response)
