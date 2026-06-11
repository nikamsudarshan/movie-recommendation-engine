# 🎬 Hybrid Movie Recommendation Engine

An end-to-end Machine Learning pipeline that generates personalized movie recommendations by merging Natural Language Processing (NLP) with behavioral matrix factorization. 

## 📌 Project Overview
This project solves the "cold start" and "filter bubble" problems inherent in standard recommendation systems by utilizing a **Hybrid Architecture**. It calculates independent scores for both contextual metadata and user interaction history, blending them into a final weighted recommendation queue.

## ⚙️ The Architecture

The engine is built on two distinct algorithms:

1. **Content-Based Filtering (TF-IDF & Cosine Similarity):** - **How it works:** Processes unstructured text data (genres, movie overviews).
   - **Method:** Converts text into a mathematical vector space using Term Frequency-Inverse Document Frequency (TF-IDF) and calculates the Cosine Similarity between movie profiles.
   
2. **Collaborative Filtering (K-Nearest Neighbors):** - **How it works:** Analyzes pure behavioral data and user rating patterns.
   - **Method:** Transforms raw interaction logs into a sparse user-item matrix and applies the K-Nearest Neighbors (KNN) algorithm to identify item similarity based on clustering.

3. **The Hybrid Merger:**
   - **How it works:** Cross-references the mathematical outputs of both models.
   - **Method:** Applies an adjustable weight (e.g., 50% Content / 50% Collaborative) to generate a unified `hybrid_score`, allowing the system to be tuned dynamically.

## 📊 Dataset
This project uses the official [MovieLens (Small) Dataset](https://grouplens.org/datasets/movielens/), containing:
- 100,000+ ratings
- 9,000+ movies
- 600+ unique users

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (TF-IDF, KNN, Cosine Similarity)
* **Environment:** Google Colab / Jupyter Notebook

## 🚀 How to Run

1. Clone this repository:
```bash
git clone [https://github.com/nikamsudarshan/movie-recommendation-engine.git](https://github.com/nikamsudarshan/movie-recommendation-engine.git)
cd movie-recommendation-engine
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt

```


3. Open the Jupyter Notebook:
```bash
jupyter notebook movie_engine.ipynb

```


4. Run all cells to download the dataset, process the NLP pipeline, and execute the recommendation engine.

