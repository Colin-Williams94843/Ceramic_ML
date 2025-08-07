
# Codes and Data for 2025 Paper on Using Machine Learning to Classify Complex Ceramic AM Samples

This repository contains the Python scripts used to generate the results in the 2025 paper on using machine learning to classify complex ceramic additive manufacturing (AM) samples.

**Citation:**  
*(Insert citation here once published)*

## Scripts

### 1. **RF_OneType.py**
   This script allows for classification of data consisting only of simulated or experimental data.

   - **`--cutoff`**: Exclude the first `n` features from the analysis (default: `0`).
   - **`--test_size`**: The percentage of data used for testing. The remaining data is used for training (default: `0.95`).
   - **`--iterations`**: Number of random seeds/splits to shuffle the data for testing stability (default: `101`).

   #### Example Usage:
   ```bash
   python RF_OneType.py --path "/your/data/path" --features_file "Experiments_13_99TallestPeaks.txt" --labels_file "Experiments_13_Labels.txt" --cutoff 0 --test_size 0.2 --iterations 5
   ```

---

### 2. **RF_Cutoff_Range.py**
   This script is used when combining synthetic and experimental data for Random Forest. Simulations are used in training, alongside a small portion of experimental data, to test on remaining unseen experimental data.

   - **`--feature_start`**: Specify the first of the 99 total features to include, so that all lower features are excluded (default: `0`).
   - **`--feature_end`**: Specify the upper-bound of the 99 total features to include in the analysis (default: `none`).
   - **`--iterations`**: Number of random seeds/splits to shuffle the data for testing stability (default: `101`).
   - **`--exp_train_pct`**: Percentage of total experimental data to be included in training, along with all simulations (default: `0.04`).

   #### Example Usage:
   ```bash
   python3 RF_Cutoff_Range.py --path "/your/data/path" --feature_start 65 --feature_end 85 --iterations 101 --exp_train_pct 0.25
   ```

---

## License

This project is licensed under the MIT License.
