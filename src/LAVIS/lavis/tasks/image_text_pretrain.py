"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from torch.nn import BCEWithLogitsLoss, MSELoss, functional as F
from scipy import spatial
# from sentence_transformers import SentenceTransformer

from lavis.common.registry import registry
from lavis.common.utils import is_convertible_to_int
from lavis.tasks.base_task import BaseTask



@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, repetition_penalty, length_penalty, top_p, temperature, evaluate, report_metric=True, annotation_file=None, sample_id_key="image_id", caption_key="caption", split=["val"], load_gt_from_file=False, img_ids = [], use_biased_tokens=False):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.top_p = top_p
        self.temperature = temperature
        self.evaluate = evaluate

        self.report_metric = report_metric
        self.annotation_file = annotation_file
        self.sample_id_key = sample_id_key
        self.caption_key = caption_key
        assert len(split) == 1, "Only support one split for evaluation."
        self.split = split[0]
        self.load_gt_from_file = load_gt_from_file
        self.img_ids = img_ids

        self.use_biased_tokens = use_biased_tokens

        self.att_loss = BCEWithLogitsLoss()
        # self.att_loss = MSELoss()

    # def evaluation(self, model, data_loader, cuda_enabled=True):
    #     pass

    def valid_step(self, model, samples):
        results = []
        # run_cfg = slf.cfg.run_cfg
        outputs = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            top_p=self.top_p,
            temperature=self.temperature,
            use_biased_tokens=self.use_biased_tokens
        )
        img_ids = samples[self.sample_id_key]
        # for pred_caption, gt_caption, img_id, att_map_pred, att_map_gt in zip(outputs["captions"], samples["text_input"], img_ids, outputs["attention_map"], samples["attention_map"]):
        #     # att_map_pred = F.sigmoid(att_map_pred)
        #     att_loss = self.att_loss(att_map_pred.squeeze(1), att_map_gt)
        for pred_caption, gt_caption, img_id in zip(outputs["captions"], samples["text_input"], img_ids):
            # att_map_pred = F.sigmoid(att_map_pred)
            # att_loss = self.att_loss(att_map_pred.squeeze(1), att_map_gt)
            att_loss = 0

            # not all img_ids are ints
            img_id = int(img_id) if is_convertible_to_int(img_id) else img_id
            if self.img_ids and img_id not in self.img_ids: # only include specified img_ids if specified
                continue

            results.append({"pred_caption": pred_caption, "image_id": img_id, "gt_caption": gt_caption, 
                            #"att_loss": att_loss.item()
            })

        return results
    
    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )

        st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        ground_truth_embds = []
        predicted_embds = []
        att_loss = []
        import torch
        import numpy as np
        with torch.no_grad():
            for idx,res in enumerate(val_result):
                embd = st_model.encode(res["gt_caption"], show_progress_bar=False, convert_to_tensor=False)
                ground_truth_embds.append(embd)

                embd = st_model.encode(res["pred_caption"], show_progress_bar=False, convert_to_tensor=False)
                predicted_embds.append(embd)

                # att_loss.append(res["att_loss"])

        def cosine_similarity(y_trues, y_preds):
            return np.mean([
                1 - spatial.distance.cosine(y_true, y_pred) 
                for y_true, y_pred in zip(y_trues, y_preds)
            ])

        if self.report_metric:
            # metrics = self._report_metrics(
            #     eval_result_file=eval_result_file, split_name=split_name
            # )
            metrics = {
                "agg_metrics": cosine_similarity(ground_truth_embds, predicted_embds),
                # "att_loss": np.mean(att_loss)
            }
        else:
            metrics = {"agg_metrics": 0.0}
        
        return metrics
    
    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 5)
        max_len = run_cfg.get("max_len", 30)
        min_len = run_cfg.get("min_len", 1)
        repetition_penalty = run_cfg.get("repetition_penalty", 1.15)
        length_penalty = run_cfg.get("length_penalty", 0.)
        top_p = run_cfg.get("top_p", 0.9)
        temperature = run_cfg.get("temperature", 1.)
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)
        annotation_file = run_cfg.get("annotation_file", None)
        sample_id_key = run_cfg.get("sample_id_key", "image_id")
        caption_key = run_cfg.get("caption_key", "caption")
        load_gt_from_file = run_cfg.get("load_gt_from_file", False)
        split = run_cfg.get("valid_splits", ["val"])
        img_ids = run_cfg.get("img_ids", []) # evaluate only subset of imgs

        use_biased_tokens = run_cfg.get("use_biased_tokens", False)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            top_p=top_p,
            temperature=temperature,
            evaluate=evaluate,
            report_metric=report_metric,
            annotation_file=annotation_file,
            sample_id_key=sample_id_key,
            caption_key=caption_key,
            split=split,
            load_gt_from_file=load_gt_from_file,
            img_ids=img_ids,
            use_biased_tokens=use_biased_tokens
        )
