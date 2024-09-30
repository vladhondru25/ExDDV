import json
import os
from enum import Enum

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class Label(Enum):
    REAL = 0
    FAKE = 1
INPUT_NO_OF_FRAMES = 5
WIDTH = 500
HEIGHT = 400

class ExplainableDataset(Dataset):
    def __init__(self) -> None:
        csv_data_fake = pd.read_csv("/home/vhondru/vhondru/phd/biodeep/xAI_deepfake/dataset.csv")
        csv_data_fake["movie_name"] = csv_data_fake["movie_name"].map(
            lambda x: os.path.join("/media/vhondru/hdd/dp/manipulated_videos/end_to_end", x)
        )
        csv_data_fake["label"] = Label.FAKE.value

        csv_data_real = csv_data_fake.copy()
        csv_data_real["movie_name"] = csv_data_real["movie_name"].map(
            lambda x: os.path.join("/media/vhondru/hdd/dp/original_data/original/data/original_sequences/youtube/c23/videos", x)
        )
        csv_data_real["label"] = Label.REAL.value

        self.csv_data = pd.concat((csv_data_fake, csv_data_real))
        self.csv_data.reset_index(inplace=True)

    def __len__(self):
        return len(self.csv_data)
    
    def __getitem__(self, index):
        video_metadata = self.csv_data.loc[index]

        movie_name = video_metadata["movie_name"]
        text = video_metadata["text"]
        click_locations = video_metadata["click_locations"]
        label = video_metadata["label"]

        cap = cv2.VideoCapture(movie_name)
        movie_no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames_indices = np.linspace(start=0, stop=movie_no_frames-1, num=INPUT_NO_OF_FRAMES)
        frames_indices = frames_indices.astype(int)

        # Use frames on clicked locations if they exist
        if click_locations != "{}":
            click_locations = json.loads(click_locations)

            frames_with_click = list(click_locations.keys())
            frames_with_click = list(map(int, frames_with_click))
            frames_with_click.sort()

            i = 0
            while len(frames_indices) - len(frames_with_click) > 0:
                if i % 2 == 0 and frames_indices[-i-1] not in frames_with_click:
                    frames_with_click.append(frames_indices[-i-1])
                elif i % 2 != 0 and frames_indices[i] not in frames_with_click:
                    frames_with_click.append(frames_indices[i])

                i = i + 1

            frames_indices = sorted(frames_with_click)

            frames_indices = list(map(int, frames_indices))

        frames = []
        for frame_ind in frames_indices[:INPUT_NO_OF_FRAMES]:
            if frame_ind == movie_no_frames:
                frame_ind = frame_ind - 1

            # Set the current frame position to the desired frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)

            # Read the frame
            ret, frame = cap.read()

            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)

        return np.stack(frames), label


if __name__ == "__main__":
    ds = ExplainableDataset()
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)

    for batch in dl:
        frames, labels = batch

        frames = frames.to(device="cuda")
        labels = labels.to(device="cuda")

        frames = torch.permute(frames, (0,1,4,2,3))

        print(frames.shape)
        print(labels)

        break
