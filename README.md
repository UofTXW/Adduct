# MLP Adduct Prediction with SHAP Interpretation

This project implements a machine learning model for predicting adduct formation using Molecular Fingerprints (ECFP4) and provides interpretability analysis using SHAP values.

## Features

- Molecular fingerprint generation using ECFP4
- Neural network model for adduct prediction
- SHAP-based model interpretation
- Molecular fragment analysis
- Visualization of important molecular features

## Requirements

- Python 3.x
- pandas
- numpy
- rdkit
- scikit-learn
- matplotlib
- shap
- joblib

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your input data in CSV format with SMILES and target values
2. Update the input path in the script
3. Run the script:

```bash
python MLP_Adduct_Interpret(ECFP4)_SHAP.py
```

## Output

The script generates:
- Model performance metrics
- SHAP value analysis
- Molecular fragment analysis
- Visualization plots

## License

MIT License 