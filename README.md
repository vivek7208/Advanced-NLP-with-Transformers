# Advanced Natural Language Processing (NLP) with Transformers

This project offers a comprehensive exploration of advanced NLP tasks using transformer models. Transformers, a deep learning model architecture, have revolutionized the field of natural language processing due to their effectiveness in handling sequential data. The project is divided into five distinct parts, each focusing on a unique NLP task: text summarization, text classification, question answering, named entity recognition, and relationship extraction. 

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
  - [Text Summarization](#text-summarization)
  - [Text Classification](#text-classification)
  - [Question Answering](#question-answering)
  - [Named Entity Recognition](#named-entity-recognition)
  - [Relationship Extraction](#relationship-extraction)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project is a comprehensive exploration of various NLP tasks using transformer models. The tasks explored include text summarization, text classification, question answering, named entity recognition, and relationship extraction. Each task is implemented in a separate Jupyter notebook. The project showcases the power of transformer models and the ease of using the Hugging Face's Transformers library to accomplish complex NLP tasks.

## Installation

The project primarily uses Python and Jupyter notebooks. The required Python libraries are:

- Transformers
- PyTorch
- SpaCy
- NLTK
- pandas
- numpy
- matplotlib

To install all dependencies, clone the repository and run:

```bash
pip install -r requirements.txt
```

## Usage

Each Jupyter notebook in the project can be run independently. They contain both the code and explanations for each step, making them self-explanatory. The notebooks are designed to be comprehensive and detailed, allowing both beginners and experienced users to understand the workings of each NLP task and how transformer models are used in each case.

## Project Structure

The project consists of five Jupyter notebooks, each focusing on a unique NLP task:

### Text Summarization

The **Text Summarization (1_summarization.ipynb)** [![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/vivek7208/Advanced-NLP-with-Transformers/blob/master/notebooks/1_summarization.ipynb) focuses on the task of summarizing long documents into short, concise summaries. The notebook uses the Hugging Face's Transformers library and leverages the **'facebook/bart-large-cnn'** pre-trained transformer model for this task. The notebook walks you through the process of deploying this model using Amazon SageMaker, making predictions, and evaluating the results using the Rouge metric.

### Text Classification

The **Text Classification (2_text_classification.ipynb)** [![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/vivek7208/Advanced-NLP-with-Transformers/blob/master/notebooks/2_text_classification.ipynb) discusses various techniques for classifying text into predefined categories. It implements a text classification model using TensorFlow and the Amazon SageMaker SDK. The specific pre-trained model used is **'tensorflow-tc-bert-en-uncased-L-12-H-768-A-12-2'**. This model is built upon a text embedding model from TensorFlow Hub, which is pre-trained on Wikipedia and BookCorpus datasets. The notebook guides you through the process of selecting a pre-trained text classification model, running inference on the model, and deploying the model using Amazon SageMaker. The model is fine-tuned (hyperparameter optimization (HPO)) on the SST2 dataset, which comprises positive and negative movie reviews, and classifies input text as either a positive or negative movie review.

|       | No HPO    | With HPO  |
|-------|-----------|-----------|
| Accuracy  | 0.582723  | 0.953148  |
| F1 Score  | 0.735376  | 0.959596  |


### Question Answering

The **Question Answering (3_question_answering.ipynb)** [![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/vivek7208/Advanced-NLP-with-Transformers/blob/master/notebooks/3_question_answering.ipynb) demonstrates the application of transformer models in question answering tasks. The notebook uses a pre-trained model from HuggingFace, specifically, the **'distilbert-base-cased-distilled-squad'** model, and trains it on the SQuAD (Stanford Question Answering Dataset) dataset. The notebook explains how to set up a hyperparameter tuning job to improve the model's performance and how to evaluate the model's performance using the Exact Match and F1 scores.

|                           | No HPO    | With HPO  |
|---------------------------|-----------|-----------|
| Average Exact Matching Score | 0.291339  | 0.535433  |
| Average F1 Score             | 0.428837  | 0.723536  |


### Named Entity Recognition

The **Named Entity Recognition (4_entity_recognition.ipynb)** [![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/vivek7208/Advanced-NLP-with-Transformers/blob/master/notebooks/4_entity_recognition.ipynb) delves into the task of identifying and classifying named entities in a text. The notebook uses the **'en_core_web_md'** model from the SpaCy library for this task. It demonstrates how to run inference on the pre-trained named entity recognition model and defines an Amazon SageMaker Model which references the source code and specifies the container to use. The notebook guides you through the process of entity recognition in a sentence, visualizing the entities, and extracting entities from a larger text like an article.

|         | No HPO    | With HPO  |
|---------|-----------|-----------|
| Precision | 0.621406  | 0.811914  |
| Recall    | 0.647711  | 0.839227  |
| F1        | 0.634286  | 0.825345  |
| Accuracy  | 0.857885  | 0.922633  |


### Relationship Extraction

The **Relationship Extraction (5_relationship_extraction.ipynb)** [![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/vivek7208/Advanced-NLP-with-Transformers/blob/master/notebooks/5_relationship_extraction.ipynb) explores the task of identifying and classifying semantic relationships between entities in a text. The notebook employs a Relationship Extraction model built on a **'Bert-base-uncased model'** using transformers from the Hugging Face's Transformers library. The model, after fine-tuning, attaches a linear classification layer that takes a pair of token embeddings outputted by the Text Embedding model and initializes the layer parameters to random values. The fine-tuning step fine-tunes all the model parameters to minimize prediction error on the input data and returns the fine-tuned model. The data used for fine-tuning is the SemEval-2010 Task 8 dataset. This dataset is used for multi-way classification of mutually exclusive semantic relations between pairs of nominals. The notebook walks you through the process of extracting relationships using various techniques, and provides methods to visualize the relationships.

|         | No HPO    | With HPO  |
|---------|-----------|-----------|
| Accuracy | 0.167096  | 0.167096  |
| F1 Macro | 0.015071  | 0.015071  |
| F1 Micro | 0.167096  | 0.167096  |


## License

This project is licensed under the terms of the MIT license.
