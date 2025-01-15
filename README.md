#Code Breakdown



This Python script is designed to analyze melt pool dimensions, defects, and printability in Laser Powder Bed Fusion (L-PBF) processes. The script can be used in a supercomputing environment to run three different versions of the Eagar-Tsai Model (the analytical model, the pre-trained NN model, and the scaled E-T model). The melt pool dimensions are then used as inputs to defined criteria that determine the boundaries for the macroscopic defects: lack of fusion (LOF), keyholing, and balling. For lack of fusion, two criteria are evaluated. For keyholing, three criteria were evaluated, while for balling, two criteria were evaluated. A total of 12 criteria are evaluated for a single composition. In addition the composition-based criteria for balling is also evaluated for the material. 

## Key Features
- Calculates material properties using the rule of mixtures and composition-based feature vector (CBFV) using the Oliynyk dataset
- In addition to these properties, other properties are calculated
- Predicts melt pool dimensions using analytical, scaled, or neural network-based E-T models.
- 


---

## Setup

### Prerequisites
- Python 3.6+
- Required Libraries:
  ```bash
  pip install -r requirements.txt

  ```

### Usage
Run the script using:
```bash
python main.py
```

You can select the E-T model type by modifying the `e_t_model_type` variable in the `main()` function to one of the following:
- `analytical`
- `scaled`
- `NN`

---

## Functions

### 1. `ROM_THERMO(results_df)`
Calculates thermodynamic properties (molecular weight, density, melting temperature, etc.) for each composition.

### 2. `melt_pool_dimensionless(data)`
Computes dimensionless parameters for the melt pool based on input parameters and material properties.

### 3. `scaled_ET(dimensionless_df)`
Implements the scaled E-T model to predict melt pool dimensions (length, width, depth).

### 4. `ET_NN(dimensionless_df)`
Applies a neural network-based E-T model for melt pool dimension predictions.

### 5. `analytical_ET(dimensionless_df)`
Uses the analytical E-T model to compute melt pool dimensions and temperature distributions.

### 6. `keyholing_criteria(dimensionless_df)`
Defines keyholing criteria based on normalized enthalpy and geometric conditions.

### 7. `lof_criteria(dimensionless_df)`
Evaluates lack-of-fusion criteria for defect analysis.

### 8. `balling(dimensionless_df, T_amb=[288])`
Identifies balling defects based on solidification and spreading times.

### 9. `cooling_rate(dimensionless_df)`
Calculates the cooling rate of the melt pool.

---


## Sample Output

Results are saved as CSV files:
- Parameters: `parameters_<file_name>_<e_t_model_type>.csv`
- Dimensionless Data: `dimensionless_df_<file_name>_<e_t_model_type>.csv`
- E-T Model Predictions: `ET_<file_name>_<e_t_model_type>.csv`
- Final Outputs: `Package_output_<file_name>_<e_t_model_type>.csv`

---

## Example Execution Flow

1. Load and preprocess the input composition data as well as the THERMOCALC calculated properties. 
2. Generate thermodynamic properties using `ROM_THERMO()`.
3. Compute dimensionless parameters with `melt_pool_dimensionless()`.
4. Select and execute the appropriate E-T model (`analytical`, `scaled`, or `NN`).
5. Analyze defects and save results.

---

## Contributing
Contributions are welcome! If you encounter any issues or have suggestions, please email sofiasheikh@tamu.edu

---

For detailed implementation and additional features, refer to the in-code comments or the project documentation.
