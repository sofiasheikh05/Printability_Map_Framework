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
It implements the scaled E-T model referred to in Ref. 2, reducing the time needed to implement the E-T analytical thermal model. However, there are some constraints involved with using the scaled model, which I would recommend users read about before using it. 

### 4. `ET-NN`
Implements the pre-trained ET-NN model discussed in the published article associated with this GitHub repository. Inputs are retrieved from the calculated thermal properties and the user-defined values for process parameters. 

### 5. `analytical_ET`
Applies a neural network-based E-T model for melt pool dimension predictions.

### 6. `G-S Depth`
Implements the correction to depth after keyholing is identified. The formulation is discussed in depth in Ref 3. 

### 7. `keyholing_normalized`
The keyholing criteria, KH2 and KH3, defined in Ref 4 and Ref 5, are calculated so that they can be analyzed using the function keyholing. 

### 8. `cooling_rate`
Calculates the cooling rate in the L-PBF process. 

### 9. `keyholing_criteria`
All three keyholing criteria are evaluated to analyze whether the criterion defining the keyholing boundary for each boundary condition is passed or not to indicate whether keyholing occurs. 

### 10. `lof_criteria`
The LOF boundary conditions LOF2 are evaluated and analyzed for each depth data point.

### 10. `balling`
A material's susceptibility to balling is evaluated. The composition-based balling criteria formulation is discussed in Ref 6.

### 11. `hot_cracking`
Calculates the hot cracking susceptibility using the Kou criteria. 

# Boundary Conditions Used to Evaluate Macroscopic Defects:


| **Defect Type** | **Label** | **Equation** |
|-----------------|-----------|--------------|
| Lack-of-Fusion  | LOF1      | D ≤ t        |
|                 | LOF2      | (h/W)² + th/(th+D) ≥ 1 |
| Balling         | Ball1     | L/W ≥ 2.3     |
|                 | Ball2     | πW/L < √(2/3) |
| Keyholing       | KH1       | W/D ≤ 2.5     |
|                 | KH2       | ΔH/hs = AP / (π hs √(α va³)) > π T_boiling / T_liquidus |
|                 | KH3       | Ke = ηP / ((T_liquidus - T₀) πρCp √(αν r₀³)) > 6 |

The code evaluates each combination criteria, and there are 12 columns with results, with defects (including defect-free print, which is labeled as Success) labeled in these columns for each P-V point. 

---
##  Results are saved as CSV files:

## File 1. Parameters
- file format: `parameters_<file_name>_<e_t_model_type>.csv`
- outputs the process parameters defined in user inputs as well as the resulting energy density values for each point.

## File 2. E-T Model Predictions:
- file format: `ET_<file_name>_<e_t_model_type>.csv`
- Outputs the results of the E-T thermal model results along with some thermal properties calculated through the package.

  ## File 3. Material Properties:
- file format: `Material_PROP_<file_name>_<e_t_model_type>.csv`
- Outputs all the material thermal properties calculated using the package in a single file.

## File 4. Final (Main) Output:
- file format: 'Package_output_<file_name>_<e_t_model_type>.csv`
-  Composition, process parameters, melt pool geometry and each criteria combination results indicating what defect is predicted for each P-V point is outputted.

---

## Example Execution Flow

1. Load and preprocess the input composition data and the THERMOCALC calculated properties. 
2. Generate thermodynamic properties using `ROM_THERMO()`.
3. Compute dimensionless parameters with `melt_pool_dimensionless()`.
5. Analyze defects and save results.

---

## Contributing
Contributions are welcome! If you encounter any issues or have suggestions, please email sofiasheikh@tamu.edu

---

