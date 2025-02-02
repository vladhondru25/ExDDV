import copy
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict

import cv2
import numpy as np
import pandas as pd
import torch
import transformers
import torchvision.transforms as T
import ujson as json
from PIL import Image
from torch.utils.data import Dataset
from decord import VideoReader, cpu

# from .params import DataArguments

IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100
LLAVA_IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"

FACEFORENSINCS_PATH = "/home/eivor/data/dp/manipulated_videos"
DEEPFAKECHALLENGE_PATH = "/home/eivor/data/deepfake_dataset_challenge"

class Label(Enum):
    REAL = 0.0
    FAKE = 1.0
INPUT_NO_OF_FRAMES = 5
WIDTH = 500
HEIGHT = 400
IGNORE_INDEX = -100

def encode_video(video_path, max_num_frames=10):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    return frames, frame_idx

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def find_closest_values(list1, list2):
    closest_values = []
    for num in list1:
        closest = min(list2, key=lambda x: abs(x - num))  # Find the closest value in list2
        closest_values.append(closest)
    return closest_values

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        # data_args: DataArguments,
        padding=True,
    ):
        super(LazySupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        # rank0_print("Formatting inputs...Skip in lazy mode")
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.max_num_frames = data_args.max_num_frames

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        is_video = False
        num_frames = None

        processor = self.processor
        if "image" in sources:
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
           
            for image_file in image_files:
                if not os.path.exists(image_file):
                    image_file = os.path.join(image_folder, image_file)
                images.append(Image.open(image_file).convert("RGB"))

        elif "video" in sources:
            video_file = sources["video"]
            video_folder = self.data_args.image_folder

            if not os.path.exists(video_file):
                video_file = os.path.join(video_folder, video_file)

            images = encode_video(video_file, self.max_num_frames)
            
            is_video = True
            num_frames = len(images)

        else:
            images = None

        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video, num_frames=num_frames))

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

class ExplainableDataset(Dataset):
    flag_use_masking = True
    flag_draw_keypoints = False
    RADIUS = 70 # 55
    assert not (flag_draw_keypoints and flag_draw_keypoints)
    
    video_question = "What is wrong in this video?" if not flag_draw_keypoints else "What is wrong in this video? The green dot will indicate the area."

    def __init__(self, split, processor) -> None:
        csv_data_fake = pd.read_csv("/home/eivor/biodeep/xAI_deepfake/dataset.csv")
        # csv_data_fake = csv_data_fake[csv_data_fake["dataset"] == "Farceforensics++"]
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

        self.split = split
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
        
        print(f'Total data: {len(self.csv_data)}')

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
    
    @staticmethod
    def draw_keypoints(image, keypoint, color_gt=(0, 255, 0), radius=5):
        """
        Draws ground-truth and predicted keypoints on the image.
        
        Args:
            image (np.ndarray)
        """
        img_with_keypoints = image.copy()
        h,w,_ = img_with_keypoints.shape
        # Draw ground-truth keypoints
        cv2.circle(img_with_keypoints, (int(keypoint["x"]*w), int(keypoint["y"]*h)), radius, color_gt, -1)

        return img_with_keypoints
    
    @staticmethod
    def apply_mask(frame, keypoint, use_hard_mask=True, radius=75):
        _,h,w = frame.shape
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
            
            frame = frame * dilated[...,None]
            frame = frame.astype(np.uint8)
            
        return frame
    
    def __getitem__(self, index):
        video_metadata = self.csv_data.loc[index]

        movie_name = video_metadata["movie_name"]
        text = video_metadata["text"]
        click_locations = video_metadata["click_locations"]
        label = video_metadata["label"]
        image_id = video_metadata["id"]

        processor = self.processor

        images, frames_indices = encode_video(movie_name, INPUT_NO_OF_FRAMES)
        
        keypoints_kwargs = {}
        if self.flag_draw_keypoints or self.flag_use_masking:
            click_locations = json.loads(click_locations)
            click_locations_time = [int(t) for t in list(click_locations.keys())]
            closest_frame_indices = find_closest_values(frames_indices, click_locations_time)
            
            keypoints = [click_locations[str(c)] for c in closest_frame_indices]
            
        if self.flag_draw_keypoints:
            images = [self.draw_keypoints(np.array(img), kp) for img,kp in zip(images,keypoints)]
            images = [Image.fromarray(img) for img in images]
        
        if self.flag_use_masking:
            keypoints_kwargs = {"keypoints": keypoints, "keypoints_fn": self.apply_mask, "use_hard_mask": True, "radius": self.RADIUS}
            # keypoints_kwargs = {"keypoints": keypoints, "keypoints_fn": self.apply_mask, "use_hard_mask": False, "radius": 65}

        is_video = True
        num_frames = len(images)

        conversation = [
            {
                "from": "human",
                "value": f"<video>\n{self.video_question}"
            },
            {
                "from": "gpt",
                "value": text
            }
        ]
        sources = copy.deepcopy(llava_to_openai(conversation, is_video=is_video, num_frames=num_frames))
    
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
                inputs = processor(user_input, images, return_tensors='pt', **keypoints_kwargs)
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
        if self.split == "val":
            data_dict["text"] = text
        
        return data_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_image_sizes = []
        batch_texts = []

        for example in examples:
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            batch_pixel_values.append(example.get("pixel_values"))
            batch_image_sizes.append(example.get("image_sizes"))
            batch_texts.append(example.get("text"))
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)
        pixel_values = torch.cat([pv for pv in batch_pixel_values if pv is not None and pv.numel() > 0], dim=0) if any(pv is not None and pv.numel() > 0 for pv in batch_pixel_values) else None
        image_sizes = torch.cat([isize for isize in batch_image_sizes if isize is not None and isize.numel() > 0], dim=0) if any(isize is not None and isize.numel() > 0 for isize in batch_image_sizes) else None

        batch_dict = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        
        if batch_texts[0] is not None:
            batch_dict.update(text=batch_texts)

        if pixel_values is not None:
            batch_dict.update(pixel_values=pixel_values, image_sizes=image_sizes)

        return batch_dict


def replace_image_tokens(input_string, start_count=1):
    count = start_count

    if LLAVA_IMAGE_TOKEN not in input_string:
        return input_string, count

    while LLAVA_IMAGE_TOKEN in input_string:
        input_string = input_string.replace(LLAVA_IMAGE_TOKEN, f"<|image_{count}|>", 1)
        count += 1

    return input_string, count

def video_to_image_tokens(input_string, num_frames):

    frame_tokens = "\n".join([LLAVA_IMAGE_TOKEN] * num_frames)
    input_string = input_string.replace(VIDEO_TOKEN, frame_tokens)

    return input_string

def llava_to_openai(conversations, is_video=False, num_frames=None):

    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    image_count = 1  # Initialize image count here
    for conversation in conversations:
        
        if is_video:
            conversation['value'] = video_to_image_tokens(conversation["value"], num_frames)
        
        transformed_content, image_count = replace_image_tokens(conversation["value"], image_count)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data

def make_supervised_data_module(processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    # sft_dataset = LazySupervisedDataset(
    #     data_path=data_args.data_path, processor=processor, data_args=data_args
    # )
    sft_dataset = ExplainableDataset(split="train", processor=processor)
    eval_dataset = ExplainableDataset(split="val", processor=processor)
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)
    # data_collator = sft_dataset.collater

    return dict(train_dataset=sft_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)
    
    
if __name__ == "__main__":
    from transformers import AutoProcessor
    
    
    processor = AutoProcessor.from_pretrained("microsoft/Phi-3.5-vision-instruct",
        cache_dir=None, 
        padding_side='right',
        trust_remote_code=True,
        num_crops=4,
        model_max_length=131072
    )
    
    sft_dataset = ExplainableDataset(split="train", processor=processor)
    sft_dataset[0]
