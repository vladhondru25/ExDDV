import random
from PIL import Image

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import tqdm

import sys
sys.path.append("..")
from dataset import ExplainableDataset
from lavis.common.dist_utils import get_rank
from lavis.models import load_model_and_preprocess
from scipy import spatial
from sentence_transformers import SentenceTransformer


def setup_seed():
    seed = 42 + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def main():
    setup_seed()

    # loads BLIP-2 pre-trained model
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device="cuda")

    model_path = "/home/vhondru/vhondru/phd/biodeep/xAI_deepfake/src/LAVIS/lavis/output/BLIP2/Caption_xAI/20241011103/checkpoint_30.pth"
    model_obj = torch.load(model_path)
    model.load_state_dict(model_obj["model"], strict=False)

    """
    input_movie_path = "/media/vhondru/hdd/dp/manipulated_videos/end_to_end/741_M010.mp4"
    frames_indices = [75, 100, 125, 150, 175]
    cap = cv2.VideoCapture(input_movie_path)

    frames = []
    for frame_ind in frames_indices:
        # Set the current frame position to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)

        # Read the frame
        ret, frame = cap.read()

        # frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = vis_processors["eval"](frame).to("cuda")

        frames.append(frame)

    frames = torch.stack(frames)

    print(frames.unsqueeze(0).shape)    
    print(frames.unsqueeze(0).dtype)
    answer = model.generate({"image": frames.unsqueeze(0)})
    print(answer)
    """

    ground_truth = []
    predicted = []
    val_dataset = ExplainableDataset(split="val", vis_processors=vis_processors["eval"])
    for i in tqdm.tqdm(range(len(val_dataset))):
        sample = val_dataset[i]
        image = sample["image"].to("cuda")

        answer = model.generate({"image": image.unsqueeze(0)})

        ground_truth.append(sample["text_input"])
        predicted.extend(answer)

    st_model = SentenceTransformer('/media/vhondru/hdd/reverse_sd/img-to-txt/models_chckpts/all-MiniLM-L6-v2', device='cuda')

    ground_truth_embds = []
    predicted_embds = []
    with torch.no_grad():
        for idx,d in enumerate(ground_truth):
            embd = st_model.encode(d, show_progress_bar=False, convert_to_tensor=False)
            ground_truth_embds.append(embd)

        for idx,d in enumerate(predicted):
            embd = st_model.encode(d, show_progress_bar=False, convert_to_tensor=False)
            predicted_embds.append(embd)

    def cosine_similarity(y_trues, y_preds):
        return np.mean([
            1 - spatial.distance.cosine(y_true, y_pred) 
            for y_true, y_pred in zip(y_trues, y_preds)
        ])

    print(ground_truth)
    print()
    print(predicted)
    print()
    print(cosine_similarity(ground_truth_embds, predicted_embds))


if __name__ == "__main__":
    main()
