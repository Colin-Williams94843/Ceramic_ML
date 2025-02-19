# Ceramic_ML
Codes and Data for 2025 Paper on Using Machine Learning to Classify Complex Ceramic AM Samples

Here are the Python scripts used to generate the results in the 2025 paper:

(insert citation here once published)

"RF_OneType.py" allows for looking at the classification of only simulated/experimental data.

  "cutoff" is the first n modes that are excluded from analysis (default = 0)

  "test_size" is the % of data used in testing, with the remaining used for training (default = 0.95)

  "iterations" is the number of random seeds/splits to shuffle the data (default = 101)

Example implementation:

python RF_OneType.py \
    --path "/your/data/path" \
    --features_file "Experiments_65_99TallestPeaks.txt" \
    --labels_file "Experiments_65_Labels.txt" \
    --cutoff 0 \
    --test_size 0.2 \
    --iterations 5

"RF_Transfer.py" is the script used during transfer learning, where all simulations are used in training plus a small amount of experimental data, for testing on remaining unseen experiments.

Example implementation:

python RF_Transfer.py --path "/your/data/path" --cutoff 30 --iterations 101 --exp_train_pct 0.05

  "exp_train_pct" is the percentage of total experimental data to be shared in training along with all the simulations (default = 0.04)
