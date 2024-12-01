import copy
import importlib 
import json
import os
from enum import Enum
from PIL import Image

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from transformers import AutoProcessor

phi3_vision_data = importlib.import_module("Phi3-Vision-Finetune.src.training.data", "pkg.subpkg")

FACEFORENSINCS_PATH = "/media/vhondru/hdd/dp/manipulated_videos"
DEEPFAKECHALLENGE_PATH = "/media/vhondru/hdd/deepfake_dataset_challenge"


class Label(Enum):
    REAL = 0.0
    FAKE = 1.0
INPUT_NO_OF_FRAMES = 5
WIDTH = 500
HEIGHT = 400
IGNORE_INDEX = -100

class ExplainableDataset(Dataset):
    def __init__(self, split, processor) -> None:
        csv_data_fake = pd.read_csv("/home/vhondru/vhondru/phd/biodeep/xAI_deepfake/dataset.csv")
        csv_data_fake["movie_name"] = csv_data_fake.apply(self._obtain_path, axis=1)
        csv_data_fake["label"] = Label.FAKE.value

        # csv_data_real = csv_data_fake.copy()
        # csv_data_real["movie_name"] = csv_data_real["movie_name"].map(
        #     lambda x: os.path.join("/media/vhondru/hdd/dp/original_data/original/data/original_sequences/youtube/c23/videos", x)
        # )
        # csv_data_real["text"] = "There is nothing unnormal in the video."
        # csv_data_real["label"] = Label.REAL.value

        # self.csv_data = pd.concat((csv_data_fake, csv_data_real))
        self.csv_data = csv_data_fake


        self.csv_data = self.csv_data.sample(frac=1, random_state=0).reset_index()

        if split == "train":
            self.csv_data = self.csv_data[100:]
        elif split == "val":
            self.csv_data = self.csv_data[:100]
        else:
            raise Exception(f"split={split} not implemented.")

        self.csv_data.reset_index(inplace=True)

        self.transform = T.Compose([
            T.ToTensor()
        ])

        self.processor = processor

    def get_img_ids(self):
        return self.csv_data["id"].to_list()

    def _obtain_path(self, row):
        if row["dataset"] == "Farceforensics++":
            return os.path.join(FACEFORENSINCS_PATH, row["manipulation"], row["movie_name"])
        elif row["dataset"] == "DeepfakeDetection":
            return os.path.join(DEEPFAKECHALLENGE_PATH, row["manipulation"], row["movie_name"])

    def __len__(self):
        return len(self.csv_data)
    
    def collater(self, samples):
        # Filter out None samples
        samples = [s for s in samples if s is not None]
        # Check if samples is empty after filtering
        if not samples:
            return {}
        collated_dict = {}
        keys = samples[0].keys() # Use the keys of the first sample as a reference
        for k in keys:
            values = [sample[k] for sample in samples]
            # If the value type for the key is torch.Tensor, stack them else return list
            collated_dict[k] = torch.stack(values, dim=0) if isinstance(values[0], torch.Tensor) else values

        return collated_dict
    
    def __getitem__(self, index):
        video_metadata = self.csv_data.loc[index]

        movie_name = video_metadata["movie_name"]
        text = video_metadata["text"]
        click_locations = video_metadata["click_locations"]
        label = video_metadata["label"]
        image_id = video_metadata["id"]

        processor = self.processor

        images = phi3_vision_data.encode_video(movie_name, INPUT_NO_OF_FRAMES)

        is_video = True
        num_frames = len(images)

        conversation = [
            {
                "from": "human",
                "value": "<video>\nWhat is wrong in this video?"
            },
            {
                "from": "gpt",
                "value": text
            }
        ]
        sources = copy.deepcopy(phi3_vision_data.llava_to_openai(conversation, is_video=is_video, num_frames=num_frames))
    
        all_input_ids = [torch.tensor([1])] # bos token id
        all_labels = [torch.tensor([-100])] # ignore bos token
        all_pixel_values = []
        all_image_sizes = []

        for idx, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_input = processor.tokenizer.apply_chat_template([user_input], tokenize=False, add_generation_prompt=True)
            gpt_response = f"{gpt_response['content']}<|end|>\n"
            
            if idx == 0:
                inputs = processor(user_input, images, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs.get('pixel_values'))
                all_image_sizes.append(inputs.get('image_sizes'))

            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        all_input_ids.append(torch.tensor([32000]))  # eos token id
        all_labels.append(torch.tensor([32000]))  # eos token id
        
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        pixel_values = torch.cat([pv for pv in all_pixel_values if pv is not None and pv.numel() > 0], dim=0) if any(pv is not None and pv.numel() >0 for pv in all_pixel_values) else None
        image_sizes = torch.cat([isize for isize in all_image_sizes if isize is not None and isize.numel() > 0], dim=0) if any(isize is not None and isize.numel()>0 for isize in all_image_sizes) else None

        attention_mask = (input_ids > -1000000).to(torch.long)
    
        data_dict = dict(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return data_dict


if __name__ == "__main__":
    MODEL_ID = "microsoft/Phi-3.5-vision-instruct"
    CACHE_DIR = None
    NUM_CROPS = 16
    MAX_SEQ_LENGTH = 131072

    processor = AutoProcessor.from_pretrained(MODEL_ID,
                                              cache_dir=CACHE_DIR, 
                                              padding_side='right',
                                              trust_remote_code=True,
                                              num_crops=NUM_CROPS,
                                              model_max_length=MAX_SEQ_LENGTH)
    
    # use unk rather than eos token to prevent endless generation
    processor.tokenizer.pad_token = processor.tokenizer.unk_token
    processor.tokenizer.pad_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.pad_token)
    processor.tokenizer.padding_side = 'right'

    ds = ExplainableDataset("train", processor)
    # print(ds[0])
    sample = ds[0]
    print(sample["input_ids"].shape)
    print(sample["pixel_values"].shape) # torch.Size([5, 17, 3, 336, 336])   5 = INPUT_NO_OF_FRAMES; 17 = NUM_CROPS + 1
    print(sample["image_sizes"].shape)
    print(sample["attention_mask"].shape)
    print(sample["labels"].shape)
