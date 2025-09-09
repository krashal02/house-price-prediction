# 🏡 House Price Prediction (Machine Learning Project)

This project predicts house prices based on the California Housing dataset using Python and Scikit-Learn.  
It covers the full ML workflow — from preprocessing to model training and inference.

## 🔹 Project Workflow
1. **Data Preprocessing**
   - Handled missing values using `SimpleImputer`
   - Scaled numerical features with `StandardScaler`
   - Encoded categorical features with `OneHotEncoder`

2. **Model Training**
   - Used **RandomForestRegressor** (also tested Linear Regression & Decision Tree)
   - Performed train-test split with stratified sampling
   - Saved trained model and preprocessing pipeline with `joblib`

3. **Inference**
   - Loads the saved model and pipeline
   - Reads `input.csv` as new data
   - Generates predictions and saves them in `output.csv`

---

## 🔹 Files in This Repository
- `main.py` → main Python script (training + inference)
- `requirements.txt` → required Python libraries
- `input.csv` → sample input for testing
- `output.csv` → generated predictions (after inference)
- `README.md` → project documentation 

⚠️ Note: Large files (`model.pkl`, `pipeline.pkl`, full dataset) are not uploaded to GitHub.  
You can download the dataset here: https://www.kaggle.com/datasets/camnugent/california-housing-prices

## How to Run the Project

1. Clone the repo:
   git clone https://github.com/krashal02/house-price-prediction.git
   cd house-price-prediction

2. Install dependencies:
   pip install -r requirements.txt

3. Run the script:
   python main.py

- First run → trains the model
- Second run → performs inference and saves output.csv

## Author 
👤 Krashal Yaduvanshi  
📍 Jaipur, India  
💼 Aspiring Data Scientist | Full Stack Developer | ML Enthusiast
