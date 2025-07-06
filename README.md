# S2R-Wind: Semi-Supervised Regression on Wind Power Forecasting

This repository contains all the code, configuration, and results for a project focused on **semi-supervised regression** for wind power prediction using multiple techniques:

* **S2RMS (Semi-Supervised Regression via Multi-Source Co-Training)**
* **ElasticNet** and **XGBoost** baselines
* **CLUS+**, a Java-based framework for semi-supervised learning with Predictive Clustering Trees (PCTs)

All experiments are performed on a real-world wind turbine dataset (`wtbdata_245days.csv`) and are evaluated across **multiple folds**, **scales of labeled data**, and **multiple model types**.


## Experiment Overview

### Datasets

* Source: `wtbdata_245days.csv` (preprocessed and filtered to 10 turbines)
* Features include lagged power outputs and timestamp encodings
* Targets: `patv_target_1`
* 8-fold split over time; each fold contains 15 days of training and 7 of testing

### Approaches

| Method         | Type            | Description                                                 |
| -------------- | --------------- | ----------------------------------------------------------- |
| **S2RMS**      | Semi-Supervised | Iterative multi-learner co-training with triplet embeddings |
| **ElasticNet** | Supervised      | Baseline linear model with grid search over regularization  |
| **XGBoost**    | Supervised      | Baseline tree-based model with cross-validation             |
| **CLUS+**      | Semi-Supervised | Java-based PCT model with `-ssl` and `-forest` enabled      |

### Labeled Data Scales

* Experiments are repeated with **10%**, **20%**, and **70%** labeled data from each fold.

## Evaluation Metrics

Each method is evaluated on:

* **RMSE**: Root Mean Squared Error
* **MAE**: Mean Absolute Error
* **RSE**: Relative Squared Error
* **R²**: Coefficient of Determination

Results are aggregated across folds and exported to:

```bash
fold_results_summary.xlsx
```

## References

* **S2RMS**: [Liu et al., 2024](https://link.springer.com/article/10.1007/s10489-024-05686-6)
* **CLUS+**: [Petković et al., 2023](https://doi.org/10.1016/j.softx.2023.101526)

```
@article{liu2024semi,
  title={Semi-supervised regression via embedding space mapping and pseudo-label smearing},
  author={Liu, Liyan and Zhang, Jin and Qian, Kun and Min, Fan},
  journal={Applied Intelligence},
  volume={54},
  number={20},
  pages={9622--9640},
  year={2024},
  publisher={Springer}
}

@article{petkovic2023clusplus,
  title={CLUSplus: A decision tree-based framework for predicting structured outputs},
  author={Petkovi{\'c}, Matej and Levati{\'c}, Jurica and Kocev, Dragi and Breskvar, Martin and D{\v{z}}eroski, Sa{\v{s}}o},
  journal={SoftwareX},
  volume={24},
  pages={101526},
  year={2023},
  publisher={Elsevier}
}
```
