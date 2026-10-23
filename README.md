# Tree Ensemble Benchmark

This repository compares three tree based classifiers on handwritten digit data: a decision tree, a random forest, and AdaBoost with SAMME. The model code is included in the notebook, and the notebook contains the experiment steps, plots, results, and explanations.

## Repository structure

```text
tree-ensemble-benchmark/
|-- 24i-2013_abdul_BAI-5A.ipynb
|-- 24i-2013_abdul_BAI-5A_report.pdf
|-- mfeat-factors.csv
|-- figures/
|   |-- eda_class_balance.png
|   |-- eda_correlation_heatmap.png
|   |-- eda_pca_projection.png
|   |-- a1_entropy_curves.png
|   |-- a3_feature_thresholds.png
|   |-- a5_noise_drift_curves.png
|   |-- a6_pruning_curve.png
|   |-- a7_confusion_matrix.png
|   |-- a7_feature_importance.png
|   |-- b2_oob_curve.png
|   |-- b3_max_features_tradeoff.png
|   |-- b4_permutation_vs_impurity.png
|   |-- c2_adaboost_error_rounds.png
|   |-- c2_sample_weight_distribution.png
|   |-- c3_noise_rf_vs_adaboost.png
|   |-- d_learning_curves.png
|   `-- d_training_time_vs_accuracy.png
`-- README.md
```

## Files

- **`24i-2013_abdul_BAI-5A.ipynb`**: The executed Jupyter Notebook. It includes the scratch implementations, data preparation, model comparisons, checks, plots, and the written report. Cell outputs and plots are saved in the notebook.
- **`24i-2013_abdul_BAI-5A_report.pdf`**: A four page report with the main results, figures, analysis, limitations, recommendation, and an example prediction.
- **`mfeat-factors.csv`**: The data file used by the notebook. It contains 2,000 handwritten digit samples, 216 numeric features, and a digit label from 0 to 9.
- **`figures/`**: PNG charts for data exploration, model behavior, evaluation, and comparisons. The notebook also displays the charts inline.
- **`README.md`**: Overview and instructions for this repository.

## Technology

- Python 3
- NumPy for the model implementations and calculations
- pandas for reading the CSV data
- Matplotlib for charts
- scikit-learn for the labeled verification comparison in the notebook
- Jupyter Notebook for running and viewing the work

## Run the notebook

Clone the repository and move into its folder:

```bash
git clone https://github.com/AbdulAzeemHashmi/tree-ensemble-benchmark.git
cd tree-ensemble-benchmark
```

Install the Python packages:

```bash
python -m pip install numpy pandas matplotlib scikit-learn notebook
```

Start Jupyter:

```bash
jupyter notebook
```

Open `24i-2013_abdul_BAI-5A.ipynb` and run the cells from top to bottom. Keep `mfeat-factors.csv` and the `figures` folder beside the notebook. Running the experiment cell may take several minutes because it trains multiple models and repeats comparisons across several seeds.

## What is measured

The notebook compares the models using accuracy, macro-F1, per digit F1, confusion matrices, noise tests, pruning, feature importance, out of bag estimates, and learning curves. It also compares selected results with scikit-learn. Random seeds and model settings are shown in the notebook.
