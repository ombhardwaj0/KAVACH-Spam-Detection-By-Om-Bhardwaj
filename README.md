# KAVACH – Spam Message Detection

KAVACH is a machine-learning web application built with **Python** and **Streamlit**. It analyzes a text message and predicts whether it is likely to be **SPAM** or **HAM (not spam)**.

> **Dataset note:** This project uses the UCI SMS Spam Collection, which contains SMS messages. The current model is therefore trained for SMS/text-message classification, not specifically for email inbox classification.

## Features

- Classifies a message as SPAM or HAM.
- Displays the model's estimated spam probability.
- Uses TF-IDF text features and a Multinomial Naive Bayes classifier.
- Shows model evaluation metrics when `model/metrics.json` is available.
- Provides sample messages and recent scan history in the Streamlit interface.
- Uses a dark, neon-style KAVACH dashboard.

## Tech Stack

- Python
- Streamlit
- scikit-learn
- TF-IDF Vectorizer
- Multinomial Naive Bayes
- pandas
- joblib

## Project Structure

```text
Spam_Email_Detection/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── dataset/
│   └── SMSSpamCollection
└── model/
    ├── spam_model.pkl
    └── metrics.json
```

The `venv/` folder is created locally and should not be uploaded to GitHub.

## Requirements

- Python
- pip
- Git (for GitHub upload)

## Run Locally on Windows

### 1. Open the project folder

```powershell
cd "C:\Users\YOUR_NAME\Desktop\Spam_Email_Detection"
```

Replace `YOUR_NAME` with your Windows account folder name.

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the environment's Python directly:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset

Place the UCI SMS Spam Collection file named `SMSSpamCollection` inside the `dataset/` folder. The training script must be configured to read that location.

### 5. Train the model

Run this when creating or retraining the model:

```powershell
python train_model.py
```

The script should save the trained model and metrics under the `model/` folder, according to the paths configured in `train_model.py`.

### 6. Start the Streamlit application

```powershell
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

To stop the app, press `Ctrl + C` in the terminal.

## Dataset

This project uses the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), a labelled collection of SMS messages commonly categorized as:

- `ham` – legitimate message
- `spam` – unwanted message

Check the dataset's terms and attribution requirements before redistributing the dataset. Do not upload it to a public repository unless you have checked the applicable terms.

## Model Workflow

1. Load labelled SMS messages from the dataset.
2. Split the data into training and testing sets.
3. Convert message text into numerical features using TF-IDF.
4. Train a Multinomial Naive Bayes classifier.
5. Evaluate the classifier on the test set.
6. Save the trained model and metrics.
7. Load the saved model in Streamlit to classify entered text.

## Evaluation

The app can display evaluation information from `model/metrics.json`, if that file exists. Results depend on the dataset, train/test split, and training code. Accuracy is a test-set metric, not a guarantee that every new message will be classified correctly.

## Deploy on Streamlit Community Cloud

1. Push the project to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in with GitHub and choose **Create app**.
4. Select your repository and the `main` branch.
5. Set the main file path to `app.py`.
6. Deploy.

Ensure `requirements.txt` lists all packages imported by the app and that the deployed app can access the trained model file.

## Important Notes

- This is an educational machine-learning project, not a production email security service.
- Predictions can be wrong. Do not rely on the model as the only way to decide whether a message is safe.
- The current dataset contains SMS messages. Email-specific performance requires a suitable labelled email dataset and retraining.
- Do not commit passwords, API keys, private credentials, or your local `venv/` directory to GitHub.

## Author

**Om Bhardwaj**



