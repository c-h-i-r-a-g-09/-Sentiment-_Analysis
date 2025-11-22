# Twitter Sentiment Analysis using PySpark

A machine learning project for sentiment analysis on social media data (Twitter and Reddit) using Apache PySpark and various classification algorithms.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements sentiment analysis on multi-social media data using PySpark's MLlib. The system classifies text into three categories:
- **Positive sentiment** (1)
- **Neutral sentiment** (0)
- **Negative sentiment** (-1)

The project compares multiple machine learning algorithms to determine the most effective approach for sentiment classification.

## ✨ Features

- **Multi-source data processing**: Combines data from Twitter and Reddit
- **Text preprocessing pipeline**: Tokenization, stop words removal, and vectorization
- **Multiple ML algorithms**: 
  - Logistic Regression
  - Naive Bayes
  - Random Forest
  - Decision Tree
  - OneVsRest Classifier
- **Comprehensive evaluation metrics**: Accuracy, F1-Score, AUC, Log Loss, RMSE, MAE, MSE, MAPE
- **Hyperparameter tuning**: Cross-validation for optimal model selection
- **Scalable architecture**: Built on Apache Spark for handling large datasets

## 📊 Dataset

- **Total samples**: 230,436 text entries
- **Training set**: 137,023 samples (70%)
- **Test set**: 58,555 samples (30%)
- **Sources**: 
  - Reddit dataset: 192,131 comments
  - Twitter dataset: 38,305 tweets

### Sentiment Distribution
| Category | Count |
|----------|-------|
| Positive (1) | 86,224 |
| Neutral (0) | 66,446 |
| Negative (-1) | 42,908 |

## 🛠️ Technologies Used

- **Apache Spark** (PySpark 3.5.3)
- **Python 3.10**
- **Machine Learning Libraries**:
  - PySpark MLlib
  - Pandas
  - NumPy
  - Matplotlib
- **Text Processing**:
  - RegexTokenizer
  - StopWordsRemover
  - CountVectorizer / TF-IDF
  - HashingTF

## 📈 Model Performance

### Accuracy Comparison

| Model | Accuracy | F1-Score | AUC | Log Loss |
|-------|----------|----------|-----|----------|
| **Logistic Regression** | **77%** | 0.47 | 0.77 | 0.45 |
| Naive Bayes | 76% | 0.42 | 0.76 | 0.68 |
| OneVsRest | 74% | 0.48 | 0.74 | 0.46 |
| Random Forest | 25% | 0.46 | 0.25 | 0.43 |

### Best Model: Logistic Regression with Cross-Validation
- **Accuracy**: 82.3%
- **Optimized Parameters**: 
  - regularization parameter (regParam): 0.1, 0.3, 0.5
  - elasticNetParam: 0.0, 0.1, 0.2
  - 5-fold cross-validation

### Detailed Metrics (Training Set - Logistic Regression)
- **Overall Accuracy**: 85.9%
- **Weighted Precision**: 87.0%
- **Weighted Recall**: 85.9%
- **Weighted F-measure**: 85.5%
- **False Positive Rate**: 8.5%
- **True Positive Rate**: 85.9%

## 🚀 Installation

### Prerequisites
```bash
# Python 3.10 or higher
# Java 8 or higher (for Spark)
```

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/sentiment-analysis-pyspark.git
cd sentiment-analysis-pyspark

# Install dependencies
pip install pyspark==3.5.3
pip install pandas numpy matplotlib

# Or use requirements.txt
pip install -r requirements.txt
```

## 💻 Usage

### Running the Notebook
```bash
# Start Jupyter Notebook
jupyter notebook Sentimental_Analysis_Project.ipynb
```

### Running as Script
```python
from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("SentimentAnalysis") \
    .getOrCreate()

# Load your data
df = spark.read.csv("path/to/your/data.csv", header=True, inferSchema=True)

# Train model
# (Add your training code here)
```

## 📁 Project Structure
```
sentiment-analysis-pyspark/
│
├── Sentimental_Analysis_Project.ipynb  # Main notebook
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── data/
│   ├── redt_dataset.csv               # Reddit data
│   └── twtr_dataset.csv               # Twitter data
├── models/                            # Saved models
└── results/                           # Output visualizations
```

## 📊 Results

### Key Findings
1. **Logistic Regression** achieved the best performance with 77% accuracy
2. **Cross-validation** improved accuracy to 82.3%
3. **TF-IDF** features performed slightly worse (74.5% accuracy) compared to Count Vectors
4. **Random Forest** significantly underperformed, likely due to overfitting

### Visualization
The project includes comparative visualizations of all model performances across multiple metrics (see notebook for charts).

## 🔮 Future Improvements

- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Add real-time sentiment analysis capability
- [ ] Expand dataset with more recent social media posts
- [ ] Include aspect-based sentiment analysis
- [ ] Deploy as REST API
- [ ] Add sentiment trend analysis over time
- [ ] Implement ensemble methods
- [ ] Add multilingual support




## 🙏 Acknowledgments

- Dataset sources: Twitter and Reddit APIs
- PySpark MLlib documentation
- College faculty and peers for guidance

## 📧 Contact

Your Name - [Chirag Arora] (mailto:your.chiragarora1309@gmail.com)


⭐ If you found this project helpful, please consider giving it a star!
