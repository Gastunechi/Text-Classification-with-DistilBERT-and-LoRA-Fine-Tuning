# Text-Classification-with-DistilBERT-and-LoRA-Fine-Tuning


## Overview
This project focuses on **text classification** using **DistilBERT** and **LoRA (Low-Rank Adaptation)** for efficient fine-tuning of pre-trained language models. The goal is to classify movie reviews as either **positive** or **negative** using a dataset from IMDb. This project was created as a personal endeavor to enhance my skills in NLP and model fine-tuning.

## Project Objectives
- **Fine-tune a pre-trained model** (DistilBERT) using the **LoRA** technique for efficient parameter adaptation.
- **Train and evaluate** a model for binary sentiment classification.
- **Demonstrate performance** through accuracy metrics and sample predictions.

## Key Features
- **DistilBERT**: A lightweight version of BERT used for fast and accurate text classification.
- **LoRA**: Parameter-efficient fine-tuning method to reduce the need for extensive computational resources.
- **Hugging Face Transformers**: Leveraged for easy integration of pre-trained models and tokenization.
- **Datasets library**: Used to load and preprocess the IMDb dataset.
- **Evaluate library**: For calculating metrics like accuracy.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#ModelArchitecture)
- [How LoRA Works](#HowLoRAWorks)
- [Results](#results)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Installation

To run this project, you will need to install the following dependencies:

```bash
pip install datasets transformers evaluate accelerate peft
```
## Dataset
The dataset used is a subset of the IMDb movie review dataset. It contains truncated reviews labeled as positive or negative:
- Training samples: 1,000
- Validation samples: 1,000
The dataset can be loaded automatically using the following line in the code:

```python
dataset = load_dataset("shawhin/imdb-truncated")
```
## Model Architecture
The project uses the following architecture:

- Pre-trained model: DistilBERT (with two output labels)
- Fine-tuning method: LoRA with parameter-efficient tuning for select layers
- Optimizer: AdamW
- Batch size: 16
- Number of epochs: 10

## How LoRA Works
LoRA reduces the number of trainable parameters by decomposing the weight matrices in attention layers into low-rank representations. This allows fine-tuning large models efficiently without needing to update all the parameters, resulting in faster training and less memory usage.

## Results
After training the model for 10 epochs with a learning rate of 1e-3, the model achieved the following results on the validation set:

- Metric	    Value
- Accuracy	    91.5

## Acknowledgments
- [Hugging Face](https://huggingface.co/) for providing excellent tools and models for NLP.
- [LoRA paper](https://arxiv.org/abs/2106.09685) for inspiring parameter-efficient fine-tuning.
- [IMDb dataset](https://www.imdb.com/) for providing the movie reviews dataset.


## License
This project is licensed under the MIT License.
