## Library Recommender Competition

This project is part of the class Data science and machine learning at UNIL with Professor Michalis Vlachos.
Our objective was to build a personalized recommendation system that suggests books to users based on their reading history and metadata using collaborative filtering and advanced machine learning.

## Final score

**Best submission MAP@10:** `0.1516`  
**Top ranked hybrid model:** `hybrid_coldtuned_025_025_050.csv`

---

## Project structure

`interactions_train.csv`: User-book interaction data  
`items.csv`: Book metadata (title, author, subject, publisher, ISBN) [Given as a database]  
`items_augmented.csv`: Enriched metadata using the Google Books API  
`cf_scores.npy`, `tfidf_scores.npy`, `bert_scores.npy`: Precomputed similarity scores [With changes over time for bert_scores.npy]  
`candidate_pairs.csv`: Top 50 book candidates per user for reranking  
`xgb_cold_*.json`: Cold-start XGBoost models with tuned hyperparameters  
`*.csv`: Submission files with recommendations  
`sample_submission.csv`: Sample of user IDs to give our recommendation and be tested on Kaggle  

---

## Exploratory Data Analysis (EDA)
This section provides an overview and insights from the interactions and items metadata, helping guide modeling decisions for recommendation.

### Dataset Overview

| Metric                     | Value        |
|----------------------------|--------------|
| Interactions rows          | 87,047       |
| Items metadata rows        | 15,291       |
| Unique users               | 7,838        |
| Unique items               | 15,109       |

### Interactions Data

| Column | Description     | Type     |
|--------|------------------|----------|
| `u`    | User ID          | `int64`  |
| `i`    | Item ID          | `int64`  |
| `t`    | UNIX Timestamp   | `float64`|

**Sample Interaction Data:**

      u      i             t
0  4456   8581  1.687541e+09
1   142   1964  1.679585e+09
2   362   3705  1.706872e+09
3  1809  11317  1.673533e+09
4  4384   1323  1.681402e+09

**Items Metadata Summary:**

| Column        | Description             | Missing Values |
|---------------|-------------------------|----------------|
| Title         | Book title              | 0              |
| Author        | Author name             | 2,653          |
| ISBN Valid    | ISBN codes              | 723            |
| Publisher     | Publishing organization | 25             |
| Subjects      | Thematic keywords       | 2,223          |
| i             | Item ID                 | 0              |

---

Sample Items:
                                               Title               Author               Publisher
0  Classification décimale universelle       NaN                  Ed du CEFAL
1  Les interactions dans l’enseignement…  Cicurel, Francine       Didier
...

### Monthly Interactions Over Time

User activity peaked early in 2023 and declined sharply in late 2024.

![Monthly Interactions](./Monthly%20Interactions%20Over%20Time.png)

### Interactions per Item (log-scaled)

Most items received few interactions, indicating a long-tail distribution.

![Interactions per Item](./Interactions%20per%20Item%20(log-scaled).png)

### Interactions per User (log-scaled)

The majority of users interacted with only a few books.

![Interactions per User](./Interactions%20per%20User%20(log-scaled).png)

### Top 10 Publishers

Gallimard and Flammarion dominate the dataset.

![Top Publishers](./Top%2010%20Publishers.png)

### Subjects Word Cloud

Common topics include “Bandes dessinées”, “Histoire”, “France”, and “Philosophie”.

![Subjects Word Cloud](./Subject%20Word%20Cloud.png)

---

Example Merged Data
A sample of user-item interactions joined with metadata:

      u      i                        Title               Publisher        datetime
0  4456   8581  Ashes falling for the sky           Albin Michel    2023-06-23
1   142   1964  La page blanche                     Delcourt        2023-03-23
...

## Summary of the approach

### 1. **Candidate generation**
I generate top-50 candidate books for each user using:
- Collaborative Filtering (CF)
- TF-IDF content similarity on titles + subjects
- BERT-based embeddings (MiniLM)

### 2. **Blended hybrid score**
I combine the 3 models using a weighted sum:
- BERT: 50%
- TF-IDF: 25%
- CF: 25%

These weights were tuned empirically based on offline MAP@10 validation and Kaggle leaderboard scores.

### 3. **Feature engineering**
For each user-item candidate pair, I compute 13 features including:
- Score + rank from CF, TF-IDF, BERT
- User activity and item popularity
- Token overlap between user’s past subjects and the item
- Title/subject metadata lengths
- Average CF score and click rate

### 4. **Segmented ranking models**
I split users into:
- **Cold users:** fewer than 10 interactions
- **Heavy users:** 10+ interactions

A separate **XGBoost ranker** model is trained for each segment using pairwise ranking (`rank:pairwise`).

### 5. **Model tuning**
I tuned hyperparameters on cold users with grid search:
- `n_estimators`: 100, 200
- `max_depth`: 3, 5
- `learning_rate`: 0.05, 0.1

The best performing model was `xgb_cold_n200_d5_lr10.json`.

---

## Offline evaluation

Offline MAP@10 reached **0.5119** using held-out interactions for validation and reranking 50 candidates per user.

---

## Enhancements with API

### Enriched metadata (API)
Using Google Books API and ISBN, I retrieved:
- Descriptions
- Categories
- Ratings

These were embedded (MiniLM) and used in some BERT-based similarity scores.

### Streamlit app
A web UI was built using Streamlit for demo and exploration:
- Live recommendations
- Book search
- Book covers
- Genre filtering
- Confidence highlighting

---

## Files for submission

| File                                 | Description                                                                 |
|--------------------------------------|-----------------------------------------------------------------------------|
| `hybrid_coldtuned_025_025_050.csv`  | Best submission (MAP@10 = 0.1516)                                           |
| `Library_recommendation_project.ipynb` | The notebook to run everything that has been done                         |
| `candidate_pairs.csv`               | Top 50 book candidates per user for reranking                              |
| `items.csv`                         | Book metadata (title, author, subject, publisher, ISBN) [Given as a database] |
| `items_augmented.csv`              | Enriched metadata using the Google Books API                               |
| `sample_submission.csv`            | Sample of user IDs to generate recommendation submissions for Kaggle        |
| `xgb_cold_n200_d5_lr10.json`       | Best cold-start model (XGBoost Ranker trained on enriched cold users)      |
| `interactions_train.csv`           | User-book interaction data                                                  |

---

## Technologies

- Python 3.11
- NumPy / Pandas / TQDM
- XGBoost
- MiniLM (for embeddings)
- Google Books API (for data augmentation)
- Streamlit (UI demo)
