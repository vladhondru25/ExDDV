# Fine-tuning Phi3-Vision Series

This repository contains a script for training the [Phi3-Vision model](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) and [Phi3.5-Vision model](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)

## Other projects

**[[Qwen2-VL Finetuning]](https://github.com/2U1/Qwen2-VL-Finetune)**<br>
**[[Llama3.2-Vision Finetuning]](https://github.com/2U1/Llama3.2-Vision-Ft)**<br>
**[[Molmo Finetune]](https://github.com/2U1/Molmo-Finetune)**<br>
**[[Pixtral Finetune]](https://github.com/2U1/Pixtral-Finetune)**

## News

- [2024/10/08] 🎉 **This fine-tuning code for Phi-3 and 3.5 Vision models is now included in the [Microsoft Phi3 Cookbook](https://github.com/microsoft/phi-3cookbook)** as a recommended example in the Fine-Tuning section.

## Update

- [2024/11/05] Added memory efficient 8-bit training.
- [2024/10/08] Demo code supports video and multi-image input.
- [2024/10/07] 🔥Supports text-only data.
- [2024/09/11] 🔥Supports video data.
- [2024/08/28] Saving non_lora_weights in checkpoint.
- [2024/08/22] 🔥Supports Phi3.5-Vision.
- [2024/07/26] 🔥Supports training vision_model with lora.
- [2024/07/16] 🔥Feature update for setting different lr in projector and vision_model.
- [2024/07/03] Added WebUI demo.
- [2024/06/27] 🔥Supports multi-image training and inference.
- [2024/06/27] Supports saving the model into safetensor.

## Table of Contents

- [Fine-tuning Phi3-Vision Series](#fine-tuning-phi3-vision-series)
  - [Other projects](#other-projects)
  - [Update](#update)
  - [Table of Contents](#table-of-contents)
  - [Supported Features](#supported-features)
  - [Installation](#installation)
    - [Using `requirements.txt`](#using-requirementstxt)
    - [Using `environment.yaml`](#using-environmentyaml)
  - [Dataset Preparation](#dataset-preparation)
  - [Training](#training)
    - [Full Finetuning](#full-finetuning)
    - [Finetune with LoRA](#finetune-with-lora)
    - [Train with video dataset](#train-with-video-dataset)
      - [Merge LoRA Weights](#merge-lora-weights)
      - [Issue for libcudnn error](#issue-for-libcudnn-error)
  - [Inference](#inference)
    - [CLI Inference](#cli-inference)
    - [Gradio Infernce (WebUI)](#gradio-infernce-webui)
  - [TODO](#todo)
  - [Known Issues](#known-issues)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)

## Supported Features

- Deepspeed
- LoRA, QLoRA
- Full-finetuning
- Enable finetuning `img_projector` and `vision_model` while using LoRA.
- Disable/enable Flash Attention 2
- Multi-image training and inference
- Video-data training
- Selecting Phi3-vision and Phi3.5-Vision

## Installation

Install the required packages using either `requirements.txt` or `environment.yml`.

### Using `requirements.txt`

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Using `environment.yaml`

```bash
conda env create -f environment.yaml
conda activate phi3v
pip install flash-attn --no-build-isolation
```

**Note:** You should install the `flash-attn` after running other libraries with `requirements.txt` or `environment.yaml`.

## Dataset Preparation

The script requires a dataset formatted according to the LLaVA specification. The dataset should be a JSON file where each entry contains information about conversations and images. Ensure that the image paths in the dataset match the provided `--image_folder`.<br>

**When using a multi-image dataset, the image tokens should all be `<image>`, and the image file names should have been in a list.**
**Please see the example below and follow format your data.**

<details>
<summary>Example for single image dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": "000000033471.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      },
      {
        "from": "human",
        "value": "What feature can be seen on the back of the bus?"
      },
      {
        "from": "gpt",
        "value": "The back of the bus features an advertisement."
      },
      {
        "from": "human",
        "value": "Is the bus driving down the street or pulled off to the side?"
      },
      {
        "from": "gpt",
        "value": "The bus is driving down the street, which is crowded with people and other vehicles."
      }
    ]
  }
  ...
]
```

</details>

<details>
<summary>Example for multi image dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": ["000000033471.jpg", "000000033472.jpg"],
    "conversations": [
      {
        "from": "human",
        "value": "<image>\n<image>\nIs the perspective of the camera differnt?"
      },
      {
        "from": "gpt",
        "value": "Yes, It the perspective of the camera is different."
      }
    ]
  }
  ...
]
```

</details>

<details>
<summary>Example for video dataset</summary>

```json
[
  {
    "id": "sample1",
    "video": "sample1.mp4",
    "conversations": [
      {
        "from": "human",
        "value": "<video>\nWhat is going on in this video?"
      },
      {
        "from": "gpt",
        "value": "A man is walking down the road."
      }
    ]
  }
  ...
]
```

**Note:** Phi3-Vision uses a video as a sequential of images.

</details>

## Training

**Note:** Freezing LLM would only work without LoRA (including vision_model LoRA).<br>
**Note:** With the mixed-dataset (e.g. some data in a batch have images while some don't) It only supports with zero2.

To run the training script, use the following command:

### Full Finetuning

```bash
bash scripts/finetune.sh
```

### Full Finetuning with 8-bit

```bash
bash scripts/finetune_8bit.sh
```

This script will finetune the model with 8bit-adamw and fp8 model dtype. If you run out of vram, you could use this.

### Finetune with LoRA

If you want to train only the language model with LoRA and perform full training for the vision model:

```bash
bash scripts/finetune_lora.sh
```

If you want to train both the language model and the vision model with LoRA:

```bash
bash scripts/finetune_lora_vision.sh
```

**IMPORTANT:** If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together.

<details>
<summary>Training arguments</summary>

- `--deepspeed` (str): Path to DeepSpeed config file (default: "scripts/zero2.json").
- `--data_path` (str): Path to the LLaVA formatted training data (a JSON file). **(Required)**
- `--image_folder` (str): Path to the images folder as referenced in the LLaVA formatted training data. **(Required)**
- `--model_id` (str): Path to the Phi3-vision model. **(Required)**
- `--output_dir` (str): Output directory for model checkpoints (default: "output/test_train").
- `--num_train_epochs` (int): Number of training epochs (default: 1).
- `--per_device_train_batch_size` (int): Training batch size per GPU per forwarding step.
- `--gradient_accumulation_steps` (int): Gradient accumulation steps (default: 4).
- `--freeze_vision_tower` (bool): Option to freeze vision_model (default: False).
- `--freeze_llm` (bool): Option to freeze LLM (default: False).
- `--tune_img_projector` (bool): Option to finetune img_projector (default: True).
- `--num_lora_modules` (int): Number of target modules to add LoRA (-1 means all layers).
- `--vision_lr` (float): Learning rate for `vision_tower` and spatial merging layer.
- `--projector_lr` (float): Learning rate for `img_projection`.
- `--learning_rate` (float): Learning rate for language module.
- `--bf16` (bool): Option for using bfloat16.
- `--lora_namespan_exclude` (str): Exclude modules with namespans to add LoRA.
- `--max_seq_length` (int): Maximum sequence length (default: 128K).
- `--num_crops` (int): Maximum crop for large size images (default: 16)
- `--max_num_frames` (int): Maxmimum frames for video dataset (default: 10)
- `--bits` (int): Quantization bits (default: 16).
- `--disable_flash_attn2` (bool): Disable Flash Attention 2.
- `--report_to` (str): Reporting tool (choices: 'tensorboard', 'wandb', 'none') (default: 'tensorboard').
- `--logging_dir` (str): Logging directory (default: "./tf-logs").
- `--lora_rank` (int): LoRA rank (default: 128).
- `--lora_alpha` (int): LoRA alpha (default: 256).
- `--lora_dropout` (float): LoRA dropout (default: 0.05).
- `--logging_steps` (int): Logging steps (default: 1).
- `--dataloader_num_workers` (int): Number of data loader workers (default: 4).

**Note:** The learning rate of `vision_model` should be 10x ~ 5x smaller than the `language_model`. <br>

</details>

### Train with video dataset

You can train the model using a video dataset. However, Phi3-Vision processes videos as a sequence of images, so you’ll need to select specific frames and treat them as multiple images for training. You can set LoRA configs and use for LoRA too.

```bash
bash scripts/finetune_video.sh
```

**Note:** When training with multiple images, setting `num_crops` to 4 typically yields better performance than 16. Additionally, you should adjust `max_num_frames` based on the available VRAM.

If you run out of vram, you can use [zero3_offload](./scripts/zero3_offload.json) instead of [zero3](./scripts/zero3_offload.json). However, using zero3 is preferred.

#### Merge LoRA Weights

```
bash scripts/merge_lora.sh
```

**Note:** Remember to replace the paths in `finetune.sh` or `finetune_lora.sh` with your specific paths. (Also in `merge_lora.sh` when using LoRA.)

#### Issue for libcudnn error

```
Could not load library libcudnn_cnn_train.so.8. Error: /usr/local/cuda-12.1/lib/libcudnn_cnn_train.so.8: undefined symbol: _ZN5cudnn3cnn34layerNormFwd_execute_internal_implERKNS_7backend11VariantPackEP11CUstream_stRNS0_18LayerNormFwdParamsERKNS1_20NormForwardOperationEmb, version libcudnn_cnn_infer.so.8
```

You could run `unset LD_LIBRARY_PATH` for this error.
You could see this [issue](https://github.com/andimarafioti/florence2-finetuning/issues/2)

## Inference

**Note:** You should use the merged weight when trained with LoRA.

### CLI Inference

```
python -m src.serve.cli \
 --model-path /path/to/merged/weight \
 --image-file /Path/to/image1, /Path/to/image2, ...
```

You can set some other generation configs like `repetition_penalty`, `temperature` etc. <br>
You can also set video too (The max_frame is set to 10. You can set this by passing argument.).

### Gradio Infernce (WebUI)

1. Install gradio

```
pip install gradio
```

2. Launch app

```
python -m src.serve.app \
    --model-path /path/to/merged/weight
```

You can launch gradio based demo with this command. This can also set some other generation configs like `repetition_penalty`, `temperature` etc.<br>
You can also set the max_frame for sampling the frames in the video. Default is set to 10.

## TODO

- [x] Saving in safetensor
- [x] Supporting multi-image training and inference.
- [x] Demo with WebUI
- [x] Support Phi3.5-vision
- [x] Support for video data
- [ ] Fast infernece with [fast-gpt](https://pytorch.org/blog/accelerating-generative-ai-2/)

## Known Issues

- [libcudnn issue](#issue-for-libcudnn-error)
- Does not support text-only data.

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

## Citation

If you find this repository useful in your project, please consider giving a :star: and citing:

```bibtex
@misc{phi3vfinetuning2023,
  author = {Gai Zhenbiao and Shao Zhenwei},
  title = {Phi3V-Finetuning},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/GaiZhenbiao/Phi3V-Finetuning},
  note = {GitHub repository},
}

@misc{phi3-vision-ft,
  author = {Yuwon Lee},
  title = {Phi-3-vision-ft},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/2U1/Phi3-Vision-ft},
  note = {GitHub repository, forked and developed from \cite{phi3vfinetuning2023}},
}
```

## Acknowledgement

This project is based on

- [LLaVA](https://github.com/haotian-liu/LLaVA): An amazing open-source project of LMM.
- [Mipha](https://github.com/zhuyiche/llava-phi): Open-source projcet of SMM with amazing capabilites.
- [Microsoft Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct): Awesome pretrained SMM using phi3.
- [Phi3V-Finetuning](https://github.com/GaiZhenbiao/Phi3V-Finetuning): Open-source project for finetuning phi-3-vision.
