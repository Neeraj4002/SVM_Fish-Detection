## Fish Disease Detection from Water quality parameters using SVM 🐟

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

A machine learning-based Streamlit web application for predicting Tilapia fish disease risk using IoT water quality parameters like **Dissolved Oxygen** and **pH**. This app supports training, evaluating, saving, and making real-time predictions.

---

## 💥 Features

- 📤 Upload Excel dataset with timestamped IoT water quality data
- 🔍 Train an SVM classifier using grid search with hyperparameter tuning
- 📈 Visualize model metrics: Accuracy, Precision, Recall, F1-score
- 🔥 Confusion matrix and decision boundary plots
- 💾 Save and load trained models
--

## 📊 Dataset Assumptions

Your Excel file should contain the following columns:
- `Datetime`
- `Dissolved Oxygen (mg/L)`
- `pH`
- `Disease Occurrence (Cases)`

A binary target `disease` is derived internally as:  

df['disease'] = (df['Disease Occurrence (Cases)'] > 1.5).astype(int)
## 📦 Installation
Clone the repository
```
git clone https://github.com/your-username/fish-disease-prediction.git
cd fish-disease-prediction
```
Create a virtual environment

```
python -m venv venv
source venv/bin/activate      # For Linux/macOS
venv\Scripts\activate         # For Windows
```
Install dependencies
```
pip install -r requirements.txt
```
Run the Streamlit app
```
streamlit run frontend.py
```
📁 Project Structure
```
.
├── core.py               # Core ML logic (data processing, training, evaluation)
├── frontend.py           # Streamlit frontend UI
├── model.pkl             # Saved trained model (generated after training)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```
---
## 💡 How to Use
1. Upload your IoT dataset (Excel) via the sidebar

2. Configure test window size and SVM parameters (C, gamma, kernel)

3. Click “Run Model” to train and evaluate

4. Visualize performance and decision boundaries

5. Click “Save Model” to persist the best model

6. Switch to the “Predict” tab to use manual inputs for disease prediction


![image](https://github.com/user-attachments/assets/415407df-5e27-4de3-89cc-bed826347413)

![image](https://github.com/user-attachments/assets/46826676-0a02-4360-af51-0f6566bfdd67)

![image](https://github.com/user-attachments/assets/335bede5-ae0b-43d4-be74-676c54df68d4)
