# Marketing-Campaign-Response-Predictor
This project builds a machine learning model that predicts whether a customer will respond to a marketing offer. It includes data exploration, model training, API development with FastAPI, containerization with Docker, and cloud deployment using Fly.io.

## Project Overview
Marketing campaigns often rely on reaching a large audience, many of whom may not respond to the offer. Sending campaigns to uninterested customers wastes money, resources, and reduces overall campaign efficiency. The challenge is to identify which customers are likely to respond to a marketing offer based on their past behavior, demographic profile, and purchasing patterns.
This project develops a predictive model that estimates the probability of a customer responding to a marketing campaign. Using historical customer data, including purchases, previous campaign responses, demographics, and engagement metrics, multiple models were trained and evaluated (Linear Regression, Random Forest, XGBoost). The best-performing model (XGBoost) predicts the likelihood of a positive response for each customer.

How it will be used:
1. Businesses can target only those customers most likely to respond, saving money and increasing campaign ROI.
2. The model is deployed as a FastAPI web service, allowing users to send customer data and receive a response probability and a binary recommendation (likely to respond or not).
3. This solution enables data-driven decision-making in marketing strategy, turning raw customer data into actionable insights.

## Dataset
The project uses the [Customer Marketing Campaign dataset]([url](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign)) containing customer demographics, purchase behaviour, and previous campaign acceptance. This repository also contains the dataset in csv.

## Key Features
- End-to-end ML pipeline
- EDA + feature engineering
- Model training using Linear Regression, Random Forest, and XGBoost, with hyperparameter tuning and selection of the best-performing model (XGBoost)
- FastAPI web service (/response endpoint)
- Docker-ready application
- Cloud deployment support (Fly.io)

## Installation
**1. Clone the repository**
git clone <your-repo-url>
cd marketing-campaign-ml

**2. Create & activate environment**
uv venv
source .venv/bin/activate

**3. Install dependencies**
uv pip install -r requirements.txt

or (if using pyproject)

uv pip install .

## Local Development
**Run the API**
uvicorn response:app --host 0.0.0.0 --port 9696 --reload

**Test the API**
python test.py

You can also explore the interactive docs at:
http://localhost:9696/docs

## Docker Usage
**Build the image**
docker build -t response-prediction .

## Run the container
docker run -p 9696:9696 response-prediction

### Prediction Endpoint
POST /response

### Request body
Customer data (validated via Pydantic).


## Technologies Used
- Python 3.12+
- FastAPI
- XGBoost
- Scikit-learn
- Pydantic
- Docker
- Uvicorn
- Fly.io
