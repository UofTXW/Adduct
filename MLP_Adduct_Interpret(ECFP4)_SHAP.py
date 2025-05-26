import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import os
import time
import shap

# -----------------------------
# Configuration
# -----------------------------
input_path = 'C:/1Xiaoyun/MLP_Adducts/NA_log_2.csv'
output_dir = 'C:/1Xiaoyun/MLP_Adducts/MLP+SHAP_3'
os.makedirs(output_dir, exist_ok=True)
fp_type = 'ECFP4'
n_bits = 2048
RANDOM_SEED = 42
BEST_HIDDEN_LAYER_SIZES = (500, 200, 500)

# -----------------------------
# Fingerprint Function
# -----------------------------
def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits))

# -----------------------------
# Load Data & Fingerprint Conversion
# -----------------------------
df = pd.read_csv(input_path)
df.columns = ['SMILES', 'MNa_MH']
df = df.dropna()
df['MNa_MH'] = df['MNa_MH'].astype(float)

# Convert to fingerprints
df['fps'] = df['SMILES'].apply(smiles_to_fingerprint)
df = df[df['fps'].notnull()]
X = np.stack(df['fps'].values)
y = df['MNa_MH'].values

# Train/Test Split
X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
    X, y, df['SMILES'].values, test_size=0.2, random_state=RANDOM_SEED
)

# -----------------------------
# Final Model Training
# -----------------------------
final_model = MLPRegressor(
    hidden_layer_sizes=BEST_HIDDEN_LAYER_SIZES,
    alpha=1e-4,
    learning_rate_init=0.001,
    batch_size=256,
    max_iter=1000,
    early_stopping=True,
    n_iter_no_change=30,
    validation_fraction=0.1,
    random_state=RANDOM_SEED,
    verbose=True
)

start = time.time()
final_model.fit(X_train, y_train)
end = time.time()

# -----------------------------
# Plotting Loss Curves
# -----------------------------
plt.figure(figsize=(10, 6))
plt.plot(final_model.loss_curve_, label='Training Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'loss_curves_final.png'), dpi=300)
plt.show()

# -----------------------------
# Evaluate Final Model Performance
# -----------------------------
y_pred = final_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Calculate relative error
relative_error = np.abs((y_test - y_pred) / y_test) * 100
mean_relative_error = np.mean(relative_error)
median_relative_error = np.median(relative_error)

# Results DataFrame
results_df = pd.DataFrame({
    'SMILES': smiles_test,
    'True': y_test,
    'Pred': y_pred,
    'Abs_Error': np.abs(y_test - y_pred),
    'Relative_Error (%)': relative_error
})
results_df.to_csv(os.path.join(output_dir, 'predictions_final.csv'), index=False)

# Performance metrics
performance_df = pd.DataFrame([{
    'R2': r2,
    'RMSE': rmse,
    'Mean_Relative_Error (%)': mean_relative_error,
    'Median_Relative_Error (%)': median_relative_error
}])
performance_df.to_csv(os.path.join(output_dir, 'performance_metrics_final.csv'), index=False)

# Print Metrics
print(f"\nTest R²: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Mean Relative Error: {mean_relative_error:.2f}%")
print(f"Median Relative Error: {median_relative_error:.2f}%")
print(f"Training Time: {(end - start)/60:.2f} min")

# Predicted vs True plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('True MNa/MH')
plt.ylabel('Predicted MNa/MH')
plt.title('True vs Predicted MNa/MH')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'prediction_scatter_final.png'), dpi=300)
plt.show()

# -----------------------------
# Model Interpretability Analysis
# -----------------------------
print("\nPerforming interpretability analysis...")

# Create directories for each interpretation method
interpret_dir = os.path.join(output_dir, 'interpretability')
shap_dir = os.path.join(interpret_dir, 'shap_analysis')
frag_dir = os.path.join(interpret_dir, 'molecular_fragments')

for directory in [interpret_dir, shap_dir, frag_dir]:
    os.makedirs(directory, exist_ok=True)

# -----------------------------
# 1. SHAP Analysis
# -----------------------------
print("Computing SHAP values...")

# Create SHAP explainer for MLP model
# Use a subset of data for background to speed up computation
background = shap.kmeans(X_train, 100)  # Use 100 samples as background
explainer = shap.KernelExplainer(final_model.predict, background)
shap_values = explainer.shap_values(X_test)

# Save SHAP results
# Summary plot
shap.summary_plot(shap_values, X_test,
                  feature_names=[f"Bit_{i}" for i in range(n_bits)],
                  show=False)
plt.savefig(os.path.join(shap_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(shap_dir, 'shap_summary.svg'), bbox_inches='tight')
plt.close()

# Bar plot
shap.summary_plot(shap_values, X_test,
                  feature_names=[f"Bit_{i}" for i in range(n_bits)],
                  plot_type="bar",
                  show=False)
plt.savefig(os.path.join(shap_dir, 'shap_importance_bar.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(shap_dir, 'shap_importance_bar.svg'), bbox_inches='tight')
plt.close()

# Save SHAP values
shap_importance = pd.DataFrame({
    'Feature': [f"Bit_{i}" for i in range(n_bits)],
    'SHAP_Importance': np.abs(shap_values).mean(0)
}).sort_values('SHAP_Importance', ascending=False)
shap_importance.to_csv(os.path.join(shap_dir, 'shap_importance.csv'), index=False)

# -----------------------------
# 2. Enhanced Feature Analysis
# -----------------------------
def get_ecfp4_details(smiles, bit_idx, radius=2, n_bits=2048):
    """
    Get molecular fragment information for a specific ECFP4 fingerprint position
    
    Args:
        smiles: SMILES string of the molecule
        bit_idx: Feature position index to analyze
        radius: ECFP4 radius, default is 2
        n_bits: Fingerprint length, default is 2048
    
    Returns:
        list: List of atomic environment information for this position
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Get Morgan fingerprint detailed information
    bit_info = {}
    _ = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bit_info)
    
    # Return None if the position doesn't exist
    if bit_idx not in bit_info:
        return None
    
    # Collect all atomic environment information for this position
    environments = []
    for atom_idx, radius_info in bit_info[bit_idx]:
        # Get environment around the center atom
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius_info, atom_idx)
        if env is not None:
            # Get atoms in the environment
            amap = {}
            submol = Chem.PathToSubmol(mol, env, atomMap=amap)
            if submol is not None:
                # Record environment information
                environments.append({
                    'center_atom_idx': atom_idx,
                    'radius': radius_info,
                    'submol_smiles': Chem.MolToSmiles(submol),
                    'center_atom_symbol': mol.GetAtomWithIdx(atom_idx).GetSymbol()
                })
    
    return environments

def analyze_top_features(shap_values, smiles_list, smiles_train=None, n_top=15, n_fragments=3):
    """
    Analyze molecular fragments corresponding to top SHAP value features
    
    Args:
        shap_values: SHAP values array
        smiles_list: List of test set molecule SMILES strings
        smiles_train: List of training set molecule SMILES strings
        n_top: Number of top features to analyze
        n_fragments: Number of top fragments to record for each feature
        
    Returns:
        list: Detailed information about top features
    """
    # Calculate mean absolute SHAP values
    mean_shap_importance = np.abs(shap_values).mean(0)
    
    # Get indices and scores of top n_top features
    top_indices = np.argsort(mean_shap_importance)[-n_top:][::-1]
    top_scores = mean_shap_importance[top_indices]
    
    # Analyze each important feature
    feature_details = []
    for idx, score in zip(top_indices, top_scores):
        # Find molecules with highest contributions to this feature
        mol_contributions = shap_values[:, idx]
        
        # Get top n_fragments positive and negative contributors
        pos_contributors = np.argsort(mol_contributions)[-n_fragments:][::-1]
        neg_contributors = np.argsort(mol_contributions)[:n_fragments]
        
        # Collect fragments from test set
        test_fragments = []
        for mol_idx in pos_contributors:
            envs = get_ecfp4_details(smiles_list[mol_idx], idx)
            if envs is not None:
                test_fragments.extend([{
                    'smiles': smiles_list[mol_idx],
                    'contribution': mol_contributions[mol_idx],
                    'type': 'positive',
                    'environments': envs
                }])
        
        for mol_idx in neg_contributors:
            envs = get_ecfp4_details(smiles_list[mol_idx], idx)
            if envs is not None:
                test_fragments.extend([{
                    'smiles': smiles_list[mol_idx],
                    'contribution': mol_contributions[mol_idx],
                    'type': 'negative',
                    'environments': envs
                }])
        
        # If no fragments found in test set and training set is provided, search there
        train_fragments = []
        if len(test_fragments) == 0 and smiles_train is not None:
            # Here you would need to compute SHAP values for training set
            # For simplicity, we'll just search for the bit in training set
            for train_smiles in smiles_train[:100]:  # Limit search to first 100 training molecules
                envs = get_ecfp4_details(train_smiles, idx)
                if envs is not None:
                    train_fragments.extend([{
                        'smiles': train_smiles,
                        'contribution': None,  # SHAP value not computed
                        'type': 'training',
                        'environments': envs
                    }])
        
        feature_details.append({
            'Feature_Position': idx,
            'SHAP_Importance': score,
            'Normalized_Importance': score / np.sum(top_scores),
            'Test_Fragments': test_fragments,
            'Train_Fragments': train_fragments
        })
    
    return feature_details

# Add to main code:
print("\nAnalyzing top 15 important features...")
feature_details = analyze_top_features(shap_values, smiles_test, smiles_train, n_top=15, n_fragments=3)

# Save results
results = []
for feature in feature_details:
    # Add test set fragments
    for frag in feature['Test_Fragments']:
        for env in frag['environments']:
            results.append({
                'Feature_Position': feature['Feature_Position'],
                'SHAP_Importance': feature['SHAP_Importance'],
                'Normalized_Importance': feature['Normalized_Importance'],
                'Representative_SMILES': frag['smiles'],
                'Contribution_Type': frag['type'],
                'Contribution_Value': frag['contribution'],
                'Fragment_SMILES': env['submol_smiles'],
                'Center_Atom': env['center_atom_symbol'],
                'Radius': env['radius'],
                'Source': 'test'
            })
    
    # Add training set fragments if no test fragments found
    if len(feature['Test_Fragments']) == 0:
        for frag in feature['Train_Fragments']:
            for env in frag['environments']:
                results.append({
                    'Feature_Position': feature['Feature_Position'],
                    'SHAP_Importance': feature['SHAP_Importance'],
                    'Normalized_Importance': feature['Normalized_Importance'],
                    'Representative_SMILES': frag['smiles'],
                    'Contribution_Type': frag['type'],
                    'Contribution_Value': 'N/A',
                    'Fragment_SMILES': env['submol_smiles'],
                    'Center_Atom': env['center_atom_symbol'],
                    'Radius': env['radius'],
                    'Source': 'train'
                })
    
    # If no fragments found at all
    if len(feature['Test_Fragments']) == 0 and len(feature['Train_Fragments']) == 0:
        results.append({
            'Feature_Position': feature['Feature_Position'],
            'SHAP_Importance': feature['SHAP_Importance'],
            'Normalized_Importance': feature['Normalized_Importance'],
            'Representative_SMILES': 'N/A',
            'Contribution_Type': 'N/A',
            'Contribution_Value': 'N/A',
            'Fragment_SMILES': 'No fragment found',
            'Center_Atom': 'N/A',
            'Radius': 'N/A',
            'Source': 'none'
        })

# Convert to DataFrame and save
results_df = pd.DataFrame(results)
results_file = os.path.join(output_dir, 'top_15_features_analysis.csv')
results_df.to_csv(results_file, index=False)

print("\nFeature analysis results:")
for feature in feature_details:
    print(f"\nFeature Position {feature['Feature_Position']}:")
    print(f"SHAP Importance: {feature['SHAP_Importance']:.4f}")
    
    if len(feature['Test_Fragments']) > 0:
        print("\nTest Set Fragments:")
        for frag in feature['Test_Fragments']:
            print(f"\nMolecule: {frag['smiles']}")
            print(f"Contribution Type: {frag['type']}")
            print(f"Contribution Value: {frag['contribution']:.4f}")
            print("Molecular Environments:")
            for env in frag['environments']:
                print(f"- Center Atom: {env['center_atom_symbol']}, "
                      f"Radius: {env['radius']}, "
                      f"Fragment: {env['submol_smiles']}")
    
    if len(feature['Train_Fragments']) > 0:
        print("\nTraining Set Fragments:")
        for frag in feature['Train_Fragments']:
            print(f"\nMolecule: {frag['smiles']}")
            print("Molecular Environments:")
            for env in frag['environments']:
                print(f"- Center Atom: {env['center_atom_symbol']}, "
                      f"Radius: {env['radius']}, "
                      f"Fragment: {env['submol_smiles']}")
    
    if len(feature['Test_Fragments']) == 0 and len(feature['Train_Fragments']) == 0:
        print("No fragments found in test or training sets")
    
    print("-" * 50)

print(f"\nDetailed results saved to: {results_file}")

def draw_molecular_fragments(feature_details, output_dir):
    """
    Draw molecular fragments for the top 15 features
    
    Args:
        feature_details: List of feature details containing molecular fragments
        output_dir: Directory to save the images
    """
    fragments_dir = os.path.join(output_dir, 'top_15_fragments')
    os.makedirs(fragments_dir, exist_ok=True)
    
    for feature in feature_details:
        feature_pos = feature['Feature_Position']
        
        # Draw fragments from test set
        for i, frag in enumerate(feature['Test_Fragments']):
            for j, env in enumerate(frag['environments']):
                submol = Chem.MolFromSmiles(env['submol_smiles'])
                if submol is not None:
                    # Save as PNG
                    img = Draw.MolToImage(submol, size=(300, 300))
                    img.save(os.path.join(fragments_dir, 
                                        f'feature_{feature_pos}_frag_{i}_{j}_{frag["type"]}.png'))
                    
                    # Save as SVG
                    svg = Draw.MolToSVG(submol, width=300, height=300, kekulize=True)
                    with open(os.path.join(fragments_dir, 
                                         f'feature_{feature_pos}_frag_{i}_{j}_{frag["type"]}.svg'), 'w') as f:
                        f.write(svg)

# Add to main code after feature analysis:
print("\nDrawing top 15 molecular fragments...")
draw_molecular_fragments(feature_details, output_dir)
print("Molecular fragments have been saved to the output directory in PNG and SVG formats.") 