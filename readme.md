# 🩺 Diabetes Prediction Project

A beginner-friendly machine learning project that predicts the likelihood of diabetes using the Pima Indians Diabetes Database. This project demonstrates a complete ML workflow from data exploration to model deployment.

---

## 📋 Project Overview

This project aims to predict diabetes occurrence using various patient health metrics. The workflow includes data preprocessing, model training, evaluation, and comparison of multiple classification algorithms. 

---

## 🎯 Goals

- Demonstrate end-to-end machine learning workflow
- Compare performance of multiple classification algorithms
- Provide clear visualizations and interpretable results
- Create a foundation for more advanced diabetes prediction models

---

## 🛠️ Tech Stack

- **Python** 3.8+
- **Data Manipulation**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Environment**: Jupyter Notebook

---

## 📁 Project Structure

```
diabetes-prediction/
├── diabetes_prediction.ipynb    # Main notebook with complete workflow
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── diabetes.csv                 # Dataset file
└── results/                     # Generated outputs
    ├── feature_histograms.png
    ├── correlation_heatmap.png
    ├── model_accuracy_comparison.png
    ├── roc_curves.png
    └── best_model.pkl
```

---

## 📊 Dataset

**Source**: Pima Indians Diabetes Database from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

**Features**:
- `Pregnancies`: Number of times pregnant
- `Glucose`: Plasma glucose concentration
- `BloodPressure`: Diastolic blood pressure (mm Hg)
- `SkinThickness`: Triceps skin fold thickness (mm)
- `Insulin`: 2-Hour serum insulin (mu U/ml)
- `BMI`: Body mass index (weight in kg/(height in m)^2)
- `DiabetesPedigreeFunction`: Diabetes pedigree function
- `Age`: Age in years

**Target**: `Outcome` (0 = No diabetes, 1 = Diabetes)

---

## 🚀 Getting Started

---

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd diabetes-prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Download `diabetes.csv` from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
   - Place it in the project root directory

---

### Running the Project

1. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open and run the notebook**
   - Open `diabetes_prediction.ipynb`
   - Run cells sequentially from top to bottom
   - Results will be saved in the `results/` directory

---

## 📈 Model Performance

The project implements and compares three machine learning algorithms:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | 77.9% | 0.717 | 0.611 | 0.66 | 0.818 |
| **K-Nearest Neighbors** | 75.3% | 0.660 | 0.611 | 0.635 | - |
| **Logistic Regression** | 70.8% | 0.600 | 0.500 | 0.545 | 0.813 |

---

## 📊 Key Visualizations

The notebook generates several insightful visualizations:

- **Feature Distribution Histograms**: Understanding data distributions
- **Correlation Heatmap**: Feature relationships and multicollinearity
- **Model Accuracy Comparison**: Side-by-side performance metrics
- **ROC Curves**: Model discrimination capability
- **Feature Importance Plot**: Most influential predictors

---