# Text Sentiment Analysis with Python

Simple text sentiment analysis experiment based on the tutorial [Getting Started with Sentiment Analysis using Python](https://huggingface.co/blog/sentiment-analysis-python). (I mostly just copied the code and adapted it to run on my machine instead of Google Colab)

## Setup

Creating a virtual environment is recommended.

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`

   Note: PyTorch may need to be installed separately. On Windows, the following command should work:

   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   ```

3. Run the `train.py` script to train the model and evaluate it on the test set.

Checkpoint files will be saved in the `results` directory.

## Results

Training was done locally on an NVIDIA RTX 3070 Ti GPU.

### IMDB Dataset

The IMDB dataset was used to train a simple classifier. The sentiment of a movie review is either positive or negative.

Below is a table of a couple of test runs I did. The training dataset size is the number of samples used for training. For evaluation, a test set equivalent to 10% of the training dataset size was used.

Run | Training Dataset Size | Epochs | Accuracy   | f1-score | Training Time |
--- | --------------------- | ------ | ---------- | -------- | ------------- |
1   | 3000                  | 2      | 0.867      | 0.867    | 01:54         |
2   | 3000                  | 2      | 0.873      | 0.876    | 02:10         |
3   | 10000                 | 8      | 0.918      | 0.917    | 24:00         |

### Emotion Dataset

The emotion dataset was used to train a classifier that can predict the emotion of a text. The emotions are: anger, fear, joy, love, sadness, and surprise.

Batch size used for training was 64.

Run | Training Dataset Size | Epochs | Accuracy   | f1-score | Training Time |
--- | --------------------- | ------ | ---------- | -------- | ------------- |
1   | Auto                  | 8      | 0.937      | 0.936    | 10:43         |

The labels in the dataset are LABEL_0, LABEL_1, etc., which isn't very human-readable. The following table shows the mapping between the labels and the emotions:

Original Label | Emotion
-------------- | -------
LABEL_0        | sadness
LABEL_1        | joy
LABEL_2        | love
LABEL_3        | anger
LABEL_4        | fear
LABEL_5        | surprise

## Using the Model

The trained model can be used to predict the sentiment of a text. The following code snippet can be used to do so:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="path/to/model/directory", tokenizer="bert-base-uncased")
prediction = classifier("This is a great example input!")

print(prediction)
```

## References

1. [Getting Started with Sentiment Analysis using Python](https://huggingface.co/blog/sentiment-analysis-python) - Federico Pascual, Hugging Face Blog
2. [bhadreshpsavani/ExploringSentimentalAnalysis](https://github.com/bhadreshpsavani/ExploringSentimentalAnalysis) - Bhadresh Savani, GitHub
