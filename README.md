#Code Breakdown

This Python script is designed for analyzing melt pool dimensions, defects, and printability in Laser Powder Bed Fusion (LPBF) processes. The script has been refactored to enhance readability and usability by following Python best practices, modularization, and adding helpful comments.

## Key Features
- Predicts melt pool dimensions using analytical, scaled, or neural network-based E-T models.
- Calculates thermodynamic properties such as density, melting temperature, and boiling temperature.
- Supports parallel processing for efficient computation.
- Includes detailed logging for monitoring execution progress and errors.

---

## Setup

### Prerequisites
- Python 3.7+
- Required Libraries:
  ```bash
  pip install numpy pandas scipy keras scikit-learn pymatgen mendeleev matplotlib
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

## Modularized Functions

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

## Enhancements

1. **Improved Readability**
   - Added clear comments and docstrings to each function.
   - Organized imports and eliminated redundancies.

2. **Error Handling**
   - Added `try-except` blocks to handle common errors (e.g., missing files, invalid inputs).

3. **Parallel Processing**
   - Utilized `concurrent.futures.ThreadPoolExecutor` for efficient computation.

4. **Logging**
   - Included logging for tracking progress and debugging.

5. **Parameter Configurations**
   - Simplified parameter configurations using dictionaries and modularized setups.

---

## Sample Output

Results are saved as CSV files:
- Parameters: `parameters_<file_name>_<e_t_model_type>.csv`
- Dimensionless Data: `dimensionless_df_<file_name>_<e_t_model_type>.csv`
- E-T Model Predictions: `ET_<file_name>_<e_t_model_type>.csv`
- Final Outputs: `Package_output_<file_name>_<e_t_model_type>.csv`

---

## Example Execution Flow

1. Load and preprocess the input composition data.
2. Generate thermodynamic properties using `ROM_THERMO()`.
3. Compute dimensionless parameters with `melt_pool_dimensionless()`.
4. Select and execute the appropriate E-T model (`analytical`, `scaled`, or `NN`).
5. Analyze defects and save results.

---

## Contributing
Contributions are welcome! If you encounter any issues or have suggestions, please create an issue or submit a pull request on GitHub.

---

For detailed implementation and additional features, refer to the in-code comments or the project documentation.
