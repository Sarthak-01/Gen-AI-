# Customer Churn Prediction — User Guide

## What This Project Does

This project uses an **Artificial Neural Network (ANN)** to predict whether a bank customer is likely to **churn** (leave the bank). It takes customer details like age, credit score, account balance, and activity status, then returns a churn probability percentage.

This helps banks identify at-risk customers so they can take retention actions before losing them.

---

## Files Explained

### Core Application

| File | What It Does |
|------|--------------|
| `app.py` | Streamlit web app. Run this to open an interactive dashboard where you can input customer data and get a churn prediction. |
| `experiments.ipynb` | Jupyter notebook used for training the neural network. Open this if you want to see how the model was built, tuned, and saved. |
| `prediction.ipynb` | Jupyter notebook for running predictions manually in a notebook environment without the Streamlit UI. |

### Trained Model & Preprocessors

| File | What It Does |
|------|--------------|
| `ann_model.h5` | The trained ANN model (weights + architecture). Loaded by `app.py` and `prediction.ipynb` to make predictions. |
| `scaler.pkl` | A `StandardScaler` fitted on the training data. It normalizes numeric features (e.g., CreditScore, Age, Balance) so the model receives properly scaled inputs. |
| `label_encoder_gender.pkl` | A `LabelEncoder` that converts Gender ("Male"/"Female") into 0/1. |
| `one_hot_encoder_geo.pkl` | A `OneHotEncoder` that converts Geography ("France", "Germany", "Spain") into 3 binary columns. |

### Data & Dependencies

| File | What It Does |
|------|--------------|
| `Churn_Modelling.csv` | The dataset containing 10,000 customer records with 13 columns (features + target). |
| `requirement.txt` | Lists all Python packages needed. Run `pip install -r requirement.txt` to install them. |
| `.streamlit/config.toml` | Streamlit theme configuration — sets dark mode, colors, and fonts for `app.py`. |

---

## How To Use

### Step 1 — Install Dependencies

Open a terminal in this folder and run:

```bash
pip install -r requirement.txt
```

### Step 2 — Launch the App

```bash
streamlit run app.py
```

This opens a web page where you can:

1. Fill in customer details (Geography, Gender, Age, Credit Score, Balance, etc.)
2. Click **Predict Churn**
3. See the churn probability displayed with a progress bar and recommendation

### Step 3 — Interpret Results

- **Green (≤50%)** → Customer is likely to stay
- **Red (>50%)** → Customer is likely to churn

The result card shows the exact probability and a brief recommendation.

---

## How Prediction Works

1. You enter customer details in the form.
2. The app preprocesses the data:
   - Encodes Gender as 0/1
   - One-hot encodes Geography into 3 columns
   - Scales all numeric features using `scaler.pkl`
3. The 12-feature input vector is fed into the ANN model.
4. The model outputs a probability between 0 and 1.
5. The result is displayed with a color-coded progress bar and recommendation.

---

## Retraining the Model

To retrain or improve the model:

1. Open `experiments.ipynb` in Jupyter.
2. Modify the architecture, hyperparameters, or training data.
3. Run the cells — the notebook will save a new `ann_model.h5`.
4. Restart the Streamlit app to use the updated model.

---

## Dataset

**Source**: `Churn_Modelling.csv` (synthetic bank customer data)

**Columns**:

| Column | Description |
|--------|-------------|
| RowNumber | Row index (dropped) |
| CustomerId | Unique customer ID (dropped) |
| Surname | Customer name (dropped) |
| CreditScore | Credit score (300–900) |
| Geography | Location: France, Germany, or Spain |
| Gender | Male or Female |
| Age | Customer age (18–92) |
| Tenure | Years with the bank (0–10) |
| Balance | Account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Whether customer has a credit card |
| IsActiveMember | Whether customer is an active member |
| EstimatedSalary | Estimated annual salary |
| Exited | **Target**: 1 = churned, 0 = stayed |
