## Code Breakdown



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
05_Printability_Package.py
```

You can select the E-T model type by modifying the `e_t_model_type` variable in the `main()` function to one of the following:
- `analytical`
- `scaled`
- `NN`

---

## Main Functions

### 1. `ROM_THERMO`
Calculates thermodynamic properties (molecular weight, density, melting temperature, etc.) for each composition.

### 2. `melt_pool_dimensionless`
Computes all parameters needed for inputs to the Eagar-Tsai thermal model, including dimensionless parameters such as B and p in Ref 1, for the scaled E-T model. This marks the last function that calculates all needed to run any E-T thermal models. 

### 3. `scaled_ET`
It implements the scaled E-T model referred to in Ref. 2. It reduces the time needed to implement the E-T analytical thermal model. However, there are some constraints involved with using the scaled model, which I would refer users to read about before using it. 

### 4. `ET-NN`
Implements the pre-trained ET-NN model discussed in the published article associated with this GitHub repository. Inputs are retrieved from the calculated thermal properties and the user-defined values for process parameters. 

### 5. `analytical_ET`
Applies a neural network-based E-T model for melt pool dimension predictions.

### 6. `G-S Depth`
Implements the correction to depth after keyholing is identified. The formulation is discussed in depth in Ref 3. 

### 7. `keyholing_normalized`
Defines keyholing criteria based on normalized enthalpy and geometric conditions.

### 8. `cooling_rate`
Evaluates lack-of-fusion criteria for defect analysis.

### 9. `keyholing_criteria`
Identifies balling defects based on solidification and spreading times.

### 10. `lof_criteria`
Calculates the cooling rate of the melt pool.

### 10. `balling`
Calculates the cooling rate of the melt pool.

### 10. `hot_cracking`
Calculates the cooling rate of the melt pool.
---

### Boundary Conditions used to Evaluate Macroscopic Defects:


| **Defect Type** | **Label** | **Equation** |
|-----------------|-----------|--------------|
| Lack-of-Fusion  | LOF1      | D ≤ t        |
|                 | LOF2      | (h/W)² + th/(th+D) ≥ 1 |
| Balling         | Ball1     | L/W ≥ 2.3     |
|                 | Ball2     | πW/L < √(2/3) |
| Keyholing       | KH1       | W/D ≤ 2.5     |
|                 | KH2       | ΔH/hs = AP / (π hs √(α va³)) > π T_boiling / T_liquidus |
|                 | KH3       | Ke = ηP / ((T_liquidus - T₀) πρCp √(αν r₀³)) > 6 |



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

