import json
import os
from enum import Enum
from PIL import Image

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from lavis.processors.blip_processors import (
    BlipCaptionProcessor,
    BlipImageEvalProcessor,
    Blip2ImageTrainProcessor
)


FACEFORENSINCS_PATH = "/home/eivor/data/dp/manipulated_videos"
DEEPFAKECHALLENGE_PATH = "/home/eivor/data/deepfake_dataset_challenge"
DEPERFORENSINCS_PATH = "/home/eivor/data/ffpp/manipulated_sequences"
BIODEEPAV_PATH = "/home/eivor/data/BioDeepAV/fake/videos"
REAL_VIDEOS_PATH = "/home/eivor/data/dp/original_data/original/data/original_sequences/youtube/c23/videos"

class Label(Enum):
    REAL = 0.0
    FAKE = 1.0
INPUT_NO_OF_FRAMES = 5
WIDTH = 500
HEIGHT = 400

class ExplainableDataset(Dataset):
    use_keypoints = False
    use_hard_masking = False
    
    def __init__(self, split, vis_processors=None) -> None:
        csv_data_fake = pd.read_csv("/home/eivor/biodeep/xAI_deepfake/dataset5.csv")
        csv_data_fake["movie_name"] = csv_data_fake.apply(self._obtain_path, axis=1)
        self.csv_data = csv_data_fake

        self.csv_data = self.csv_data.sample(frac=1, random_state=0).reset_index()

        self.split = split
        if split == "train":
            self.csv_data = self.csv_data[self.csv_data["split"] == "train"]
        elif split == "val":
            self.csv_data = self.csv_data[self.csv_data["split"] == "val"]
        elif split == "test":
            self.csv_data = self.csv_data[self.csv_data["split"] == "test"]
        elif split == "train+val"
            self.csv_data = self.csv_data[self.csv_data["split"] != "test"]
        else:
            raise Exception(f"split={split} not implemented.")

        self.csv_data.reset_index(inplace=True)

        self.vis_processor = vis_processors
        self.text_processor = BlipCaptionProcessor()

        self.counter = 0
        self.users = {"Vlad Hondru": 0, "Eduard Hogea": 0}

    def get_img_ids(self):
        return self.csv_data["id"].to_list()

    def _obtain_path(self, row):
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
        # return default_collate(samples)
        
    @staticmethod
    def apply_mask(frame, keypoint, use_hard_mask=True, radius=75):
        c,h,w = frame.shape
        kp_x = keypoint["x"] * w
        kp_y = keypoint["y"] * h
        
        # Create a coordinate grid
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Compute the distance from the keypoint
        distance = np.sqrt((xx - kp_x)**2 + (yy - kp_y)**2)
        
        # Create a mask where the distance is less than or equal to the radius
        mask = (distance <= radius).astype(np.uint8)
        
        if use_hard_mask:
            frame = frame * mask[None,...]
        else:
            blur_ksize = 83
            dilation_iter = 1
            
            # Ensure binary mask is in float format for processing
            mask = mask.astype(np.float32)
            
            # Gaussian blur to create a smooth transition
            blurred = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 83)
            
            # Normalize the blurred mask to range [0, 1]
            blurred_normalized = cv2.normalize(blurred, None, 0, 1, cv2.NORM_MINMAX)
            
            # Dilation to reinforce or expand the transition
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated = cv2.dilate(blurred_normalized, kernel, iterations=dilation_iter)
            
            frame = frame * dilated[None,...]
            
        return frame
    
    def find_closest_values(self, list1, list2):
        closest_values = []
        for num in list1:
            closest = min(list2, key=lambda x: abs(x - num))  # Find the closest value in list2
            closest_values.append(closest)
        return closest_values
    
    def __getitem__(self, index):
        video_metadata = self.csv_data.loc[index]

        movie_name = video_metadata["movie_name"]
        text = video_metadata["text"]
        click_locations = video_metadata["click_locations"]
        image_id = video_metadata["id"]

        cap = cv2.VideoCapture(movie_name)
        movie_no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames_indices = np.linspace(start=0, stop=movie_no_frames-1, num=INPUT_NO_OF_FRAMES)
        frames_indices = frames_indices.astype(int)
        
        click_locations = json.loads(click_locations)
        click_locations_time = [int(t) for t in list(click_locations.keys())]
        closest_frame_indices = self.find_closest_values(frames_indices, click_locations_time)
            
        keypoints = [click_locations[str(c)] for c in closest_frame_indices]

        frames = []
        for frame_ind,keypoint in zip(frames_indices[:INPUT_NO_OF_FRAMES],keypoints[:INPUT_NO_OF_FRAMES]):
            if frame_ind == movie_no_frames:
                frame_ind = frame_ind - 1

            # Set the current frame position to the desired frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)

            # Read the frame
            ret, frame = cap.read()

            # frame = cv2.resize(frame, (WIDTH, HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = self.vis_processor(frame)

            if self.use_keypoints:
                frame = self.apply_mask(frame, keypoint, use_hard_mask=self.use_hard_masking)

            frames.append(frame)

        text_input = self.text_processor(text)
    
        return {
            "image": torch.stack(frames),
            "text_input": text_input,
            "image_id": image_id, 
        }


if __name__ == "__main__":
    """
    ds = ExplainableDataset("train", Blip2ImageTrainProcessor())
    # ds = ExplainableDataset("val", BlipImageEvalProcessor(image_size=364)),

    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)

    for batch in tqdm(dl):
        continue

    print(ds.counter)
    print(ds.users)
    """

    # ds = ExplainableDataset("train", Blip2ImageTrainProcessor())
    ds2 = ExplainableDataset("val", BlipImageEvalProcessor(image_size=364))

    dl2 = DataLoader(ds2, batch_size=8, shuffle=True, num_workers=0, pin_memory=True, collate_fn=ds2.collater)

    for batch in tqdm(dl2):
        continue

    print(ds2.counter)
    print(ds2.users)

    #     frames = batch["image"].to(device="cuda")
    #     labels = batch["label"].to(device="cuda")
    #     text_input = batch["text_input"]#.to(device="cuda")

    #     # frames = torch.permute(frames, (0,1,4,2,3))

    #     print(frames.shape)
    #     print(labels)
    #     print(text_input)

    #     break
