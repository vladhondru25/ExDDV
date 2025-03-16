import os
import json
import random
from PIL import Image

import cv2
import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import tqdm
from evaluate import load

import sys
sys.path.append("..")
from dataset import ExplainableDataset
from lavis.common.dist_utils import get_rank
from lavis.models import load_model_and_preprocess
from lavis.processors.blip_processors import BlipImageEvalProcessor
from scipy import spatial
# from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


FACEFORENSINCS_PATH = "/home/eivor/data/dp/manipulated_videos"
DEEPFAKECHALLENGE_PATH = "/home/eivor/data/deepfake_dataset_challenge"
DEPERFORENSINCS_PATH = "/home/eivor/data/ffpp/manipulated_sequences"
BIODEEPAV_PATH = "/home/eivor/data/BioDeepAV/fake/videos"
REAL_VIDEOS_PATH = "/home/eivor/data/dp/original_data/original/data/original_sequences/youtube/c23/videos"

def setup_seed():
    seed = 42 + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def main():
    # setup_seed()

    # loads BLIP-2 pre-trained model
    # model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device="cuda")
    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device="cuda")
    # model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device="cuda")
    # ask a random question.
    # ['singapore']

    # model_path = "/home/eivor/biodeep/xAI_deepfake/src/LAVIS/lavis/output/BLIP2/Pretrain_stage2/20250228173/checkpoint_9.pth"
    # model_obj = torch.load(model_path)
    # model.load_state_dict(model_obj["model"], strict=False)

    def _obtain_path(row):
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
        
    csv_data = pd.read_csv("/home/eivor/biodeep/xAI_deepfake/dataset5.csv")
    csv_data["movie_name"] = csv_data.apply(_obtain_path, axis=1)
    csv_data = csv_data[csv_data["split"] == "test"]
    csv_data.reset_index(inplace=True)

    gt_sentences = []
    pred_sentences = []
    
    for i,row in csv_data.iterrows():
        movie_name = row["movie_name"]
        text = row["text"]
        click_locations = row["click_locations"]
        image_id = row["id"]

        cap = cv2.VideoCapture(movie_name)
        movie_no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Set the current frame position to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, movie_no_frames//2)
        # Read the frame
        ret, frame = cap.read()
        raw_image = Image.fromarray(frame[...,::-1])
        cap.release()
        
        image = vis_processors["eval"](raw_image).unsqueeze(0).to("cuda")
        # generate caption
        answers = model.generate({"image": image})
        # image = vis_processors["eval"](raw_image).unsqueeze(0).to("cuda")
        # question = txt_processors["eval"](question)
        # answers = model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")
        gt_sentences.append(text)
        pred_sentences.extend(answers)
        
    # Write down predictions and gt
    json_data = []
    for gt_sentence, pred_sentence in zip(gt_sentences,pred_sentences):
        json_data.append({"gt": gt_sentence, "pred": pred_sentence})
    with open("/home/eivor/biodeep/xAI_deepfake/src/LAVIS/lavis/output/BLIP2/Pretrain_stage2/predictions_baseline.json", "w", encoding='utf-8') as out_file:
        json.dump(json_data, out_file, ensure_ascii=False, indent=4)

    # st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    # ground_truth_embds = []
    # predicted_embds = []
    # with torch.no_grad():
    #     for idx,d in enumerate(gt_sentences):
    #         embd = st_model.encode(d, show_progress_bar=False, convert_to_tensor=False)
    #         ground_truth_embds.append(embd)

    #     for idx,d in enumerate(pred_sentences):
    #         embd = st_model.encode(d, show_progress_bar=False, convert_to_tensor=False)
    #         predicted_embds.append(embd)

    def cosine_similarity(y_trues, y_preds):
        return np.mean([
            1 - spatial.distance.cosine(y_true, y_pred) 
            for y_true, y_pred in zip(y_trues, y_preds)
        ])

    # print(f"Sentence embedding cosine similarity: ", cosine_similarity(ground_truth_embds, predicted_embds))

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


if __name__ == "__main__":
    main()
