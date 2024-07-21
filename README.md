# Software Crash Detection System

This project aims to build a system that can detect software crashes by analyzing log files. The system uses a fine-tuned GPT-2 model to predict the type of error based on the log file and user feedback.

## Requirements

- Python 3.6+
- `transformers` library
- `datasets` library
- `flask` library
- `pandas` library
- `torch` library

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/software-crash-detection.git
   cd software-crash-detection
   ```

2. Install the required libraries:

    ``` bash
    pip install -r requirements.txt
    ```

## Data Preparation

Prepare your data for training:

Ensure your log data is in CSV format. Place the CSV file in the project directory.

Run the data preparation script to preprocess the data:

    ```bash
    python3 data_preparation.py
    ``` 

## Model Training

Train the GPT-2 model on your preprocessed data:

Run the training script:

    ```bash
    python3 train_model.py
    ```

This will fine-tune the GPT-2 model on your log data. The fine-tuned model will be saved in the ./fine_tuned_model directory.

## Running the Flask Application

1. Start the Flask application to handle file uploads and predictions:

Run the Flask app:

    ```bash
    python3 app.py
    ```

2. Open your web browser and navigate to http://localhost:5000.

## Usage

1. Upload a log file using the form.
2. Select the problem type from the dropdown menu.
3. Provide any additional feedback in the textbox.
4. Submit the form to get the prediction.