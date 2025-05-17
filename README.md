## Library Recommender Competition

This project is part of the class Data science and machine learning at UNIL with Professor Michalis Vlachos.
Our objective was to build a personalized recommendation system that suggests books to users based on their reading history and metadata using collaborative filtering and advanced machine learning.

## Final Score

**Best Submission MAP@10:** `0.1516`  
**Top Ranked Hybrid Model:** `hybrid_coldtuned_025_025_050.csv`

---

## Project Structure

`interactions_train.csv`: User-book interaction data  
`items.csv`: Book metadata (title, author, subject, publisher, ISBN) [Given as a database]  
`items_augmented.csv`: Enriched metadata using the Google Books API  
`cf_scores.npy`, `tfidf_scores.npy`, `bert_scores.npy`: Precomputed similarity scores [With changes over time for bert_scores.npy]  
`candidate_pairs.csv`: Top 50 book candidates per user for reranking  
`xgb_cold_*.json`: Cold-start XGBoost models with tuned hyperparameters  
`*.csv`: Submission files with recommendations  
`sample_submission.csv`: Sample of user IDs to give our recommendation and be tested on Kaggle  

---

## Summary of the Approach

### 1. **Candidate Generation**
I generate top-50 candidate books for each user using:
- Collaborative Filtering (CF)
- TF-IDF content similarity on titles + subjects
- BERT-based embeddings (MiniLM)

### 2. **Blended Hybrid Score**
I combine the 3 models using a weighted sum:
- BERT: 50%
- TF-IDF: 25%
- CF: 25%

These weights were tuned empirically based on offline MAP@10 validation and Kaggle leaderboard scores.

### 3. **Feature Engineering**
For each user-item candidate pair, I compute 13 features including:
- Score + rank from CF, TF-IDF, BERT
- User activity and item popularity
- Token overlap between user’s past subjects and the item
- Title/subject metadata lengths
- Average CF score and click rate

### 4. **Segmented Ranking Models**
I split users into:
- **Cold users:** fewer than 10 interactions
- **Heavy users:** 10+ interactions

A separate **XGBoost Ranker** model is trained for each segment using pairwise ranking (`rank:pairwise`).

### 5. **Model Tuning**
I tuned hyperparameters on cold users with grid search:
- `n_estimators`: 100, 200
- `max_depth`: 3, 5
- `learning_rate`: 0.05, 0.1

The best performing model was `xgb_cold_n200_d5_lr10.json`.

---

## Offline Evaluation

Offline MAP@10 reached **0.5119** using held-out interactions for validation and reranking 50 candidates per user.

---

## Enhancements with API

### Enriched Metadata (API)
Using Google Books API and ISBN, I retrieved:
- Descriptions
- Categories
- Ratings

These were embedded (MiniLM) and used in some BERT-based similarity scores.

### Streamlit App
A web UI was built using Streamlit for demo and exploration:
- Live recommendations
- Book search
- Book covers
- Genre filtering
- Confidence highlighting

---

## Files for Submission

| File                                | Description                                                                 |
|-------------------------------------|-----------------------------------------------------------------------------|
| `hybrid_coldtuned_025_025_050.csv` | Best submission (MAP@10 = 0.1516)                                           |
| `Library_recommendation_project.ipynb` | The notebook to run everything that has been done                         |
| `candidate_pairs.csv`              | Top 50 book candidates per user for reranking                              |
| `items.csv`                        | Book metadata (title, author, subject, publisher, ISBN) [Given as a database] |
| `items_augmented.csv`             | Enriched metadata using the Google Books API                               |
| `cf_scores.npy`, `tfidf_scores.npy`, `bert_scores.npy` | Precomputed similarity scores *(BERT scores updated over time)*  |
| `sample_submission.csv`           | Sample of user IDs to generate recommendation submissions for Kaggle        |
| `xgb_cold_n200_d5_lr10.json`      | Best cold-start model (XGBoost Ranker trained on enriched cold users)      |

---

## Technologies

- Python 3.11
- NumPy / Pandas / TQDM
- XGBoost
- MiniLM (for embeddings)
- Google Books API (for data augmentation)
- Streamlit (UI demo)
