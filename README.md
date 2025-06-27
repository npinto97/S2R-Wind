
# Semi-Supervised Regression on Wind Turbine Dataset using S2RMS

This repository contains the implementation and evaluation of the **S2RMS** semi-supervised regression framework applied to a real-world wind turbine dataset. The project replicates and extends the method proposed in the original S2RMS paper and benchmarks it against two supervised learning baselines: **ElasticNet** and **XGBoost**.

> Liu, Liyan, et al. *Semi-supervised regression via embedding space mapping and pseudo-label smearing*. Applied Intelligence, 54(20), 9622–9640 (2024). [https://doi.org/10.1007/s10489-023-04966-y](https://doi.org/10.1007/s10489-023-04966-y)

## Project Structure

```
.
├── data/                    # Processed wind turbine dataset
├── s2rmslib/                # Core implementation of S2RMS (from original repo)
├── models/                  # Saved models for each method
├── results/                 # Experimental results (CSV, JSON)
├── configs/                 # YAML configuration files
├── notebooks/               # Jupyter notebooks for analysis and plotting
├── prepare\_dataset.py       # Script to normalize and convert data to ARFF
├── run\_experiment.py        # Main script to run S2RMS on wind data
└── README.md

````

## Dataset
The dataset used in this project can be downloaded from the following link: [Download wtb_features.csv from Dropbox](https://www.dropbox.com/scl/fi/rrzuxzsz7n4u3uk69yw58/wtb_features.csv?rlkey=h8qggia4yytos2c0x6y17tj74&st=hlhywowd&dl=0)

Once downloaded, place the file inside the `data/processed/` folder before running any scripts. The preprocessing script (`prepare_dataset.py`) expects the dataset to be available in that location.

## How to Run

1. **Prepare the environment**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Preprocess the dataset**
   Converts the processed CSV data to `.arff` format with feature normalization.

   ```bash
   python prepare_dataset.py
   ```

3. **Run S2RMS experiments**

   ```bash
   python run_experiment.py
   ```

4. **Train baselines (ElasticNet & XGBoost)**
   See the corresponding scripts or notebooks in `notebooks/`.



## Results

* S2RMS is evaluated at **1%, 5%, and 10% labeled data**.
* The same 2000-instance subset is used across all models for fair comparison.
* Evaluation metric: **Root Mean Squared Error (RMSE)** on the remaining dataset.

See `notebooks/results_analysis.ipynb` for charts and result tables.