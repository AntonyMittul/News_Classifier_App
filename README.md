# 📰 News Classifier & Summarizer (Streamlit App)

## 📌 Overview
This project is a **News Article Classification and Summarization App** built with **Streamlit**.  
It classifies news articles into **4 categories** — *World, Sports, Business, and Sci/Tech* — using the **AG News dataset**.  

🔑 Features:
- Accepts **manual text input** (single/multiple news headlines).  
- Supports **PDF Upload** → Extracts text from newspapers and generates:  
  - **Headlines**  
  - **Summaries**  
  - **Top Keywords**  
  - **Predicted Category**  
- Displays **confidence scores & visualizations**.  

This project demonstrates **Natural Language Processing (NLP)** with **scikit-learn** and **transformers**, combined with a clean interactive web UI using **Streamlit**.  

---

## 📊 Dataset
The model is trained on the [AG News Classification Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset).  
- 120,000 training samples  
- 7,600 test samples  
- Categories: **World, Sports, Business, Sci/Tech**

---

## ⚙️ Tech Stack
- **Python** (scikit-learn, pandas, numpy)  
- **Streamlit** (frontend & deployment)  
- **PDF Processing**: PyPDF2, pdfplumber  
- **NLP**: TF-IDF, Logistic Regression / Transformers  
- **Keyword Extraction**: KeyBERT  

---

## 🚀 How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/news-classifier-streamlit.git
   cd news-classifier-streamlit
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Open browser → `http://localhost:8501`

---

## 📷 Features in Action
- ✅ **Text Input Classification** (single or batch headlines)  
- ✅ **PDF Upload → Extract & Classify**  
- ✅ **Confidence Score Visualization** (bar chart)  
- ✅ **Category Distribution Chart** (batch predictions)  

*(Add screenshots or GIFs here after running your app)*

---

## 📈 Future Enhancements
- Integrate **transformer models** (BERT, DistilBERT) for better accuracy.  
- Add **multilingual support** (classify non-English news).  
- Deploy on **Streamlit Cloud / Hugging Face Spaces** for public access.  

---

## 👨‍💻 Author
**S. Antony Mittul**  
📌 Final-year Computer Science student | Data Science & Analytics Enthusiast  
📍 Hindustan University, Chennai, Tamil Nadu  
