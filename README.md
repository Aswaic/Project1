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

### Sample Interaction Data

| u    | i     | t             |
|------|-------|----------------|
| 4456 | 8581  | 1.687541e+09   |
|  142 | 1964  | 1.679585e+09   |
|  362 | 3705  | 1.706872e+09   |
| 1809 | 11317 | 1.673533e+09   |
| 4384 | 1323  | 1.681402e+09   |


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

### Sample Items

| Title                                  | Author             | Publisher     |
|----------------------------------------|--------------------|---------------|
| Classification décimale universelle    | *NaN*              | Ed du CEFAL   |
| Les interactions dans l’enseignement… | Cicurel, Francine  | Didier        |


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

### Example Merged Data

A sample of user-item interactions joined with metadata:

| u    | i     | Title                        | Publisher     | datetime   |
|------|-------|------------------------------|---------------|------------|
| 4456 | 8581  | Ashes falling for the sky    | Albin Michel  | 2023-06-23 |
|  142 | 1964  | La page blanche              | Delcourt      | 2023-03-23 |

## Evaluation of Recommendations

To evaluate how well the recommendation system aligns with user interests, we selected five users and compared their top-10 recommended books with their actual reading history.

A prediction is considered **good** if it was previously interacted with by the user; otherwise, it's considered **bad**. This provides a qualitative measure of how personalized and relevant the results are.

---

### User 2226

**Clicked History:**
- Histoire des peurs alimentaires : du Moyen Age à l'aube du XXe siècle
- Le patrimoine culinaire suisse
- La condition immigrée : les ouvriers italiens en Suisse

| Rank | Recommended Title                                                                                   | Match |
|------|------------------------------------------------------------------------------------------------------|-------|
| 1    | La condition immigrée : les ouvriers italiens en Suisse                                              | ✅ Good |
| 2    | Histoire des peurs alimentaires : du Moyen Age à l'aube du XXe siècle                                | ✅ Good |
| 3    | Le patrimoine culinaire suisse                                                                       | ✅ Good |
| 4    | L'immigration en Suisse : cinquante ans d'entrouverture                                              | ❌ Bad  |
| 5    | Histoire religieuse de la Suisse : la présence des catholiques                                       | ❌ Bad  |
| 6    | La migration italienne dans la Suisse d'après-guerre                                                 | ❌ Bad  |
| 7    | Le syndicalisme suisse : histoire politique de l'Union syndicale : 1880-1980                         | ❌ Bad  |
| 8    | Femmes et discriminations en Suisse : le poids de l'histoire, XVIe - début XXe siècle                | ❌ Bad  |
| 9    | Les Suisses et l'environnement : une histoire du rapport à la nature, du XVIIIe siècle à nos jours  | ❌ Bad  |
| 10   | Monstres, démons et merveilles à la fin du Moyen Age                                                 | ❌ Bad  |

**Summary:** 3 / 10 recommendations were in the user's history.

---

### User 2177

**Clicked History:**
- La faïence de Nevers : 1585–1900
- Le parfum : des origines à nos jours

| Rank | Recommended Title                                                                 | Match |
|------|------------------------------------------------------------------------------------|-------|
| 1    | La faïence de Nevers : 1585–1900                                                   | ✅ Good |
| 2    | Le parfum : des origines à nos jours                                               | ✅ Good |
| 3    | La Belle Époque : la France de 1900 à 1914                                         | ❌ Bad  |
| 4    | La famille en France à l'époque moderne (XVIe–XVIIIe siècle)                       | ❌ Bad  |
| 5    | La révolution matérielle : une histoire de la consommation                         | ❌ Bad  |
| 6    | Journal de Jean Héroard                                                            | ❌ Bad  |
| 7    | L'éclaireur                                                                         | ❌ Bad  |
| 8    | Histoire de la France religieuse                                                   | ❌ Bad  |
| 9    | Le mariage et l'amour en France : de la Renaissance à la Révolution               | ❌ Bad  |
| 10   | Le temps des féminismes                                                            | ❌ Bad  |

**Summary:** 2 / 10 recommendations were in the user's history.

---

### User 4236

**Clicked History:**
- Hound dog
- Culottées : des femmes qui ne font que ce qu'elles veulent
- Le château des animaux
- De cape et de crocs : l'intégrale

| Rank | Recommended Title                                                   | Match |
|------|----------------------------------------------------------------------|-------|
| 1    | De cape et de crocs : l'intégrale                                   | ✅ Good |
| 2    | Le château des animaux                                              | ✅ Good |
| 3    | L'Odyssée                                                           | ❌ Bad  |
| 4    | Culottées : des femmes qui ne font que ce qu'elles veulent         | ✅ Good |
| 5    | La naissance des Dieux                                              | ❌ Bad  |
| 6    | Aya de Yopougon                                                     | ❌ Bad  |
| 7    | Bienvenue                                                           | ❌ Bad  |
| 8    | Giant                                                               | ❌ Bad  |
| 9    | On la trouvait plutôt jolie                                         | ❌ Bad  |
| 10   | Lancelot                                                            | ❌ Bad  |

**Summary:** 3 / 10 recommendations were in the user's history.

---

### User 6558

**Clicked History:**
- L'énigme de la chambre 622 : roman
- 100 idées pour mieux gérer les troubles de l'attention
- Troubles de l'attention avec ou sans hyperactivité
- La disparition de Stephanie Mailer : roman
- Plan d'études romand : cycle 1

| Rank | Recommended Title                                                                                     | Match |
|------|--------------------------------------------------------------------------------------------------------|-------|
| 1    | TDAH à l'école : petite histoire d'une inclusion                                                       | ✅ Good |
| 2    | La disparition de Stephanie Mailer : roman                                                             | ✅ Good |
| 3    | Troubles de l'attention avec ou sans hyperactivité                                                     | ✅ Good |
| 4    | 100 idées pour mieux gérer les troubles de l'attention                                                 | ✅ Good |
| 5    | L'énigme de la chambre 622 : roman                                                                     | ✅ Good |
| 6    | Plan d'études romand : cycle 1                                                                         | ✅ Good |
| 7    | Plan d'études romand : cycle 2                                                                         | ❌ Bad  |
| 8    | Le TDAH chez l'enfant et l'adolescent                                                                  | ❌ Bad  |
| 9    | Pratiques pédagogiques et TDAH : pistes de compréhension et outils pratiques                           | ❌ Bad  |
| 10   | Plan d'études romand : cycle 3                                                                         | ❌ Bad  |

**Summary:** 6 / 10 recommendations were in the user's history.

---

### User 5491

**Clicked History:**
- Histoire du monde au XIXe siècle
- L'époque contemporaine : 1770–1914
- L'émergence du monde ouvrier en Suisse au XIXe siècle

| Rank | Recommended Title                                                                             | Match |
|------|------------------------------------------------------------------------------------------------|-------|
| 1    | Histoire du monde au XIXe siècle                                                               | ✅ Good |
| 2    | L'émergence du monde ouvrier en Suisse au XIXe siècle                                          | ✅ Good |
| 3    | L'époque contemporaine : 1770–1914                                                             | ✅ Good |
| 4    | Histoire de la Suisse et des Suisses dans la marche du monde                                  | ❌ Bad  |
| 5    | La transformation du monde : une histoire globale du XIXe siècle                               | ❌ Bad  |
| 6    | Histoire et combats : mouvement ouvrier et socialisme en Suisse                                | ❌ Bad  |
| 7    | Histoire du tourisme en Suisse au XIXe siècle                                                  | ❌ Bad  |
| 8    | L'histoire du monde se fait en Asie                                                            | ❌ Bad  |
| 9    | Histoire économique de la Suisse au XXe siècle                                                 | ❌ Bad  |
| 10   | Deux siècles de luttes : une brève histoire du mouvement socialiste et ouvrier en Suisse       | ❌ Bad  |

**Summary:** 3 / 10 recommendations were in the user's history.

---

## Observations

- The recommender system successfully retrieved previously interacted items in many cases.
- In highly specialized domains (e.g., TDAH, graphic novels), recommendations were more accurate.
- Broader historical themes sometimes led to generic or tangential results.

### Recommendation Performance (Precision@10 and Recall@10)

| Method           | Precision@10 | Recall@10 |
|------------------|--------------|-----------|
| User-User CF     | 0.0061       | 0.0612    |
| Item-Item CF     | 0.0050       | 0.0498    |
| Hybrid Model     | 0.0740       | 0.7405    |

We evaluated the recommendation quality of three different models using a leave-one-out cross-validation strategy. For each user, one held-out item was used as ground truth, and Precision@10 and Recall@10 were computed over the top-10 recommended books.

Traditional collaborative filtering methods (User-User CF and Item-Item CF) showed limited effectiveness due to the sparsity and cold-start nature of the dataset. In contrast, our hybrid model, which combines CF, TF-IDF, and BERT-based similarity scores with tuned weights, significantly outperformed the baselines. The hybrid model achieved a Precision@10 of 0.0740 and an exceptionally high Recall@10 of 0.7405, confirming its ability to recover relevant books that users were likely to interact with.

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
