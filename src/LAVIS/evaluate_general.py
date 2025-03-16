import json
import sys

import numpy as np
import torch
import tqdm
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from scipy import spatial
from sentence_transformers import SentenceTransformer


INPUT_JSON_PREDICTIONS = "/home/eivor/biodeep/xAI_deepfake/src/LAVIS/lavis/output/BLIP2/Pretrain_stage2/predictions_baseline.json"


def cosine_similarity(y_trues, y_preds):
    return np.mean([
        1 - spatial.distance.cosine(y_true, y_pred) 
        for y_true, y_pred in zip(y_trues, y_preds)
    ])

def main():
    # Read input json
    with open(INPUT_JSON_PREDICTIONS) as json_file:
        json_data = json.load(json_file)
    
    # Obtain the ground-truth and predictions
    gt_sentences, pred_sentences = [], []
    for entry in json_data:
        gt_sentences.append(entry["gt"])
        pred_sentences.append(entry["pred"])
    
    # Sentence Transformer
    st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    ground_truth_embds = []
    predicted_embds = []
    with torch.no_grad():
        for idx,d in enumerate(gt_sentences):
            embd = st_model.encode(d, show_progress_bar=False, convert_to_tensor=False)
            ground_truth_embds.append(embd)

        for idx,d in enumerate(pred_sentences):
            embd = st_model.encode(d, show_progress_bar=False, convert_to_tensor=False)
            predicted_embds.append(embd)

    print(f"Sentence embedding cosine similarity: ", cosine_similarity(ground_truth_embds, predicted_embds))
    
    # BERT Score
    bertscore = load("bertscore")
    scores = bertscore.compute(predictions=pred_sentences, references=gt_sentences, lang="en", rescale_with_baseline=True, idf=True, model_type="distilbert-base-uncased")
    print(f"BERT score f1: ", np.mean(scores["f1"]))
    
    # Compute BLEU scores for each pair
    smooth = SmoothingFunction().method1
    bleu_scores = [
        sentence_bleu([gt.split()], pred.split(), smoothing_function=smooth, weights=[0.5, 0.5])
        for gt, pred in zip(gt_sentences, pred_sentences)
    ]
    average_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"Average BLEU Score: {average_bleu:.4f}") # Average BLEU Score: 0.2391

    # Compute METEOR score
    meteor_scores = [
        meteor_score([gt.split()], pred.split())
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
