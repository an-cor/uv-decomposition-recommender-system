# UV-Decomposition Recommendation System

**Data Mining Final Project | Spring 2025**

A movie recommendation project exploring **UV decomposition (latent-factor matrix factorization)** and collaborative filtering. The project includes a research report on the algorithm and an executable Jupyter notebook that trains and evaluates a **LightFM** recommender using the MovieLens 100K dataset.

> **Project scope:** This is an academic analysis and prototype, not a production recommendation service. The notebook uses LightFM's existing matrix-factorization implementation; it is not a from-scratch implementation of the UV/SGD algorithm discussed in the report.

## Project artifacts

| File | Contents |
| --- | --- |
| [`DataMiningFinal.ipynb`](DataMiningFinal.ipynb) | Python implementation, evaluation, recommendations, and visualizations |
| [`DataMiningFinal.pdf`](DataMiningFinal.pdf) | Written explanation of UV decomposition, its mathematical formulation, computational cost, experiments, and limitations |

## Problem and approach

Recommendation systems estimate a user's preferences for items they have not yet rated or interacted with. A user–item matrix is typically sparse, so latent-factor models approximate the matrix using lower-dimensional user and item representations.

The **report** explains the UV decomposition formulation:

$$
M \approx UV
$$

The **notebook** studies movie recommendation as an **implicit-feedback ranking problem**:

1. Download and load the MovieLens 100K dataset.
2. Retain ratings of **4 or 5 stars** as positive interactions.
3. Explore rating distributions and interaction sparsity.
4. Split positive interactions into training and testing sets (80/20).
5. Construct a popularity-based baseline.
6. Train a **LightFM** factorization model with **WARP ranking loss**.
7. Evaluate **Precision@10** and **Recall@10**, including users with limited histories.
8. Produce personalized movie recommendations and visualize item embeddings using **t-SNE**.

## Dataset

The source is [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/), which contains **100,000 explicit ratings** from **943 users** on **1,682 movies**.

For the notebook's ranking experiment, only ratings **>= 4** are retained as positive feedback. The saved notebook output shows:

| Split | Positive interactions |
| --- | ---: |
| Training | 44,300 |
| Test | 11,075 |
| **Total** | **55,375** |

The notebook downloads the dataset when run; the dataset is **not included** in this repository.

## Model configuration and notebook results

The notebook trains LightFM using:

| Setting | Value |
| --- | --- |
| Algorithm | LightFM with WARP loss |
| Latent dimensions (`no_components`) | 20 |
| Training epochs | 30 |
| Learning rate | 0.05 |
| User and item regularization (`user_alpha`, `item_alpha`) | `1e-6` each |
| Random state | 42 |
| Training threads | 4 |

**Results recorded in the supplied notebook:**

| Measurement | Recorded result |
| --- | ---: |
| Precision@10 (LightFM) | **0.207** |
| Recall@10 (LightFM) | **0.224** |
| Precision@10, users with fewer than 10 training interactions | **0.080** |
| Number of evaluated limited-history users | **84** |

The limited-history analysis illustrates a **cold-start / data-sparsity limitation**: recommendations are less accurate for users with few positive interactions. These are saved notebook outputs and have not been independently rerun here.

### Example recommendations

The notebook generates five new movie suggestions for a sample user and converts MovieLens item IDs into readable titles. Among the saved recommendations are:

- *The Fugitive* (1993)
- *Independence Day* (1996)
- *Die Hard* (1988)
- *Star Trek: First Contact* (1996)
- *Speed* (1994)

The notebook also uses t-SNE to project learned item factors into two dimensions for exploratory visualization.

## Report versus notebook: an important distinction

The written report additionally presents an **explicit-rating** factorization experiment using regularized squared-error optimization and user/item biases. It reports a best **RMSE of 0.93** with **32 latent factors** and **λ = 0.05**, plus comparisons with mean-rating and item-based collaborative-filtering baselines.

**That reported RMSE experiment is not implemented in the supplied notebook.** The executable notebook instead uses 20 latent factors, LightFM/WARP, and ranking metrics. The two sets of results should be read as distinct project artifacts rather than a single reproduced experiment.

The notebook also labels a popularity-baseline score of `0.062` as "Precision" while computing the number of held-out positive interactions hit by the top ten popular items **divided by all held-out positive interactions**. This denominator produces a recall-like coverage statistic, not the same per-user Precision@10 returned by LightFM. A valid quantitative comparison requires evaluating both models with a consistent metric and candidate/exclusion policy.

## Running the notebook

**Google Colab is the easiest starting point**, as the notebook was written with notebook shell commands to install dependencies and download data.

1. Open [`DataMiningFinal.ipynb`](DataMiningFinal.ipynb) in Google Colab or Jupyter.
2. Install the dependencies. In a local Python environment you can use:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Execute the cells from top to bottom. The notebook's first cells download and extract MovieLens 100K from GroupLens, so an internet connection is needed.
4. Review the printed evaluation metrics, recommendation example, and plots.

**Environment note:** LightFM includes compiled extensions. A compatible Python environment/build toolchain may be required, especially outside the original Colab environment. Exact runtimes and metrics may vary with library versions and execution environment.

## Limitations and possible improvements

- The implementation uses positive-only interactions (ratings 4–5), leaving lower ratings out of the ranking model.
- Cold-start performance declines when users have few observed positive interactions.
- Movie titles are displayed for interpretation, but content metadata is not used as model features.
- The notebook uses a single held-out split rather than repeated cross-validation.
- The popularity baseline needs a metric correction before a fair numerical comparison.
- The report's RMSE-based study and the notebook's WARP-based study would need to be aligned to reproduce one unified results table.

Potential next steps include consistent top-K baseline evaluation, additional hyperparameter sweeps, ranking metrics across multiple values of K, and hybrid models that use movie features.

## Background and references

These are **supporting readings and libraries**, not original deliverables of this repository.

- Leskovec, Rajaraman, and Ullman, *Mining of Massive Datasets*, Chapter 9: Recommendation Systems — [book website](https://www.mmds.org/).
- Koren, Bell, and Volinsky, *Matrix Factorization Techniques for Recommender Systems* (2009) — discussed in the report.
- Zhang, Yao, Sun, and Tay, *Deep Learning Based Recommender System: A Survey and New Perspectives* — [arXiv](https://arxiv.org/abs/1707.07435).
- [LightFM documentation](https://making.lyst.com/lightfm/docs/home.html) — used for model fitting and ranking evaluation.
- [MovieLens datasets](https://grouplens.org/datasets/movielens/) — data source.
- [Surprise](https://surpriselib.com/) and [scikit-optimize](https://scikit-optimize.github.io/stable/) — additional tools discussed in the report.

---

**Author:** Angel Coronel  
**Course:** Data Mining (CSC 84040)
