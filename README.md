# IMDB Sentiment Classifier

## Description

This project implements a binary sentiment classifier for IMDB movie reviews using a Long Short Term Memory (LSTM) neural network with PyTorch. It's designed to classify reviews as either positive or negative. The project includes scripts for data loading, model training, evaluation, and predictions on new text, and is also configured for easy integration with PyTorch Hub.

## Installation

Follow these steps to set up the project on your local machine.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/r4m-t1n/text-sentiment-pytorch](https://github.com/r4m-t1n/text-sentiment-pytorch)
    cd text-sentiment-pytorch
    ```

2.  **Create a Python virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: `venv\Scripts\activate`
    ```

3.  **Download the IMDB dataset:**
    You can download its .rar file manually from [here](https://ai.stanford.edu/~amaas/data/sentiment/) or from this repository on `data/` and extract it right there. The dataset structure looks like the following:
    ```
    data/
    └── aclImdb/
        ├── train/
        │   ├── pos/
        │   └── neg/
        └── test/
            ├── pos/
            └── neg/
    ```

4.  **Install dependencies:**

    * **For prediction:**
        This will install the necessary packages to load and use the pretrained model.
        ```bash
        pip install -r requirements/requirements.txt
        ```

    * **For Training (with CUDA GPU support):**
        If you plan to train the model and have a CUDA-enabled GPU, you need to install the appropriate PyTorch version with CUDA.

        * **Important:** Ensure your NVIDIA drivers and CUDA Toolkit version are compatible with the PyTorch version you are installing. You can find detailed installation instructions on the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

        * Install all training dependencies, including the CUDA-enabled PyTorch:
            ```bash
            pip install -r requirements/requirements_train.txt
            ```
            *(Note: `requirements/requirements_train.txt` includes `torch==2.3.0+cu118` with its specific index URL. Make sure your system's CUDA Toolkit version matches `cu118` or adjust the version in the file. You can also browse available PyTorch wheels here: [https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html))*

## Usage

### Training the Model

To train the sentiment classifier model:
```bash
python src/train.py
```
The training script will save the best model (based on validation accuracy) to models/sentiment_model.pt. If a previous model exists, it will resume training from the last saved epoch.

### Evaluating the Model

To evaluate the trained model on positive and negative reviews in the test set:
```
python src/evaluate.py
```

### Making Predictions on New Text

To use the trained model for predictions on new text inputs:
```
python src/predict.py
```
You will be asked to enter text. Type -1 to exit.

## PyTorch Hub Usage

You can load this model and its utility functions directly using PyTorch Hub. Ensure you have the requirements/requirements.txt dependencies installed. For further information, look at `examples/load_from_torch_hub.ipynb` folder.