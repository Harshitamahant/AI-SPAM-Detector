# 📧 AI Real-Time Spam Email Detector

An AI-powered real-time spam detection system built using Natural Language Processing (NLP), TF-IDF vectorization, and the Multinomial Naive Bayes classification algorithm.

---

## 🚀 Project Overview

This project classifies email messages as **spam** or **ham** (not spam) using a machine learning model trained on a labeled dataset. The model processes the input through the following steps:

- Text preprocessing using **Regular Expressions** and **NLTK**
- Feature extraction using **TF-IDF Vectorization**
- Model training using **Multinomial Naive Bayes**
- Real-time prediction using a basic **Python interface** or **HTML UI**

---

## 📂 Project Structure



---

## 🛠️ Tech Stack

- **Python**
- **Jupyter Notebook / Google Colab**
- **Pandas & NumPy**
- **NLTK** (Natural Language Toolkit)
- **scikit-learn**
- **TF-IDF Vectorizer**
- **Multinomial Naive Bayes**
- **ipywidgets / HTML + CSS UI (Optional)**

---

## 💡 Key Features

- 📌 Real-time email spam classification
- 🧠 Trained on real labeled dataset (`spam.csv`)
- 🧹 Text cleaning & NLP preprocessing
- 🔎 TF-IDF for meaningful feature extraction
- ✅ High performance using Naive Bayes classifier
- 🎯 Evaluation metrics: Accuracy, Precision, Recall, F1-score
- 🖼️ Optional Web UI for better presentation

---

## 🧪 Methodology

1. **Data Preprocessing**
   - Lowercasing
   - Removing punctuation, numbers, and stopwords

2. **TF-IDF Vectorization**
   - Converts text into numerical form
   - Captures word importance in the context of all emails

3. **Model Training**
   - Train-Test split (80:20)
   - Multinomial Naive Bayes classifier

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score, Confusion Matrix

5. **Deployment (optional)**
   - Interface using ipywidgets or Web UI (HTML/CSS)

---

## 📊 Results

Achieved:
- ✅ Accuracy: *~98%* on test set
- ✅ High precision and recall for spam class

---

## 🎬 How to Run

1. **Clone this repo**
   ```bash
   git clone https://github.com/yourusername/AI-SPAM-Detector.git
   cd AI-SPAM-Detector
2. Open the notebook
Run AISPAM_detect.ipynb in Jupyter or Google Colab
Upload the dataset
Upload spam.csv when prompted
Try predictions
Enter any email message in the UI box and click “Predict”

💌 Acknowledgements
SMS Spam Collection Dataset by UCI
scikit-learn & NLTK community
Open-source contributors

📷 Screenshots (Optional)
Spam detector interface

Accuracy scores
![image](https://github.com/user-attachments/assets/65aecb04-b1d2-4f53-99b8-20195ac4b60a)

Confusion matrix plot
![image](https://github.com/user-attachments/assets/18e08afe-e1d2-44f6-b274-96f4cedbf0d0)

🤝 Contributions
Contributions are welcome! Feel free to fork, clone, and submit pull requests.



