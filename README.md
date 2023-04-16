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

## Notes

All test runs were done with a batch size of 16. This was limited by the amount of memory available on my GPU.
