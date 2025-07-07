import os

scales = [10, 20, 70]
folds = range(8)
TARGET_NAME = "patv_target_1"

def generate_settings_from_arff(arff_path, test_path, settings_path, target_name=TARGET_NAME, scale=0):
    with open(arff_path, 'r') as f:
        lines = f.readlines()

    attr_lines = [line for line in lines if line.strip().lower().startswith('@attribute')]
    attr_names = [line.split()[1] for line in attr_lines]

    if target_name not in attr_names:
        raise ValueError(f"Target '{target_name}' not found in ARFF file: {arff_path}")

    target_index = attr_names.index(target_name) + 1  # CLUS uses 1-based indexing

    settings_content = f"""[General]
RandomSeed = 42

[Data]
File = {arff_path}
TestSet = {test_path}
PruneSet = None

[Attributes]
Descriptive = 1-19
Target = 20

[Tree]
Heuristic = VarianceReduction

[Ensemble]
SelectRandomSubspaces = SQRT
EnsembleMethod = RForest
FeatureRanking = Genie3
Iterations = 10 

[SemiSupervised]
SemiSupervisedMethod = PCT

[Output]
WritePredictions = Train, Test

"""

    with open(settings_path, "w") as f:
        f.write(settings_content)

    print(f"Settings written: {settings_path}")

base_data_path = "C:/Users/Ningo/Desktop/S2R-Wind/data/swdpf/clus_ssl_arffs"
settings_output_path = "C:/Users/Ningo/Desktop/S2R-Wind/data/swdpf/clus_ssl_settings"

for fold in folds:
    test_file_path = os.path.join(base_data_path, f"fold{fold}_test.arff")
    for scale in scales:
        train_file_path = os.path.join(base_data_path, f"fold{fold}_ssl_{scale}.arff")
        settings_file_path = os.path.join(settings_output_path, f"fold{fold}_ssl_{scale}.s")

        generate_settings_from_arff(train_file_path, test_file_path, settings_file_path, scale=scale)
