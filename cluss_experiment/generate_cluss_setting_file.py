import os

scales = [10, 20, 70]
folds = range(8)

def generate_settings_from_arff(arff_path, settings_path, target_name="patv_target_1"):
    with open(arff_path, 'r') as f:
        lines = f.readlines()

    attr_lines = [line for line in lines if line.strip().lower().startswith('@attribute')]
    attr_names = [line.split()[1] for line in attr_lines]

    if target_name not in attr_names:
        raise ValueError(f"Target '{target_name}' not found in ARFF file.")

    target_index = attr_names.index(target_name) + 1  # CLUS usa indici 1-based
    descriptive_range = f"1-{len(attr_names)}"

    settings_content = f"""[Data]
File = {arff_path}

[Attributes]
Descriptive = {descriptive_range}
Target = {target_index}

[Tree]
Heuristic = VarianceReduction

[Ensemble]
SelectRandomSubspaces = SQRT
EnsembleMethod = RForest
FeatureRanking = Genie3
Iterations = 10

[SemiSupervised]
SemiSupervisedMethod = PCT

[General]
RandomSeed = 42
"""

    with open(settings_path, "w") as f:
        f.write(settings_content)

    print(f"Settings file created at: {settings_path}")

for fold in folds:
    for scale in scales:
        arff_file = f"C:/Users/Ningo/Desktop/S2R-Wind/data/swdpf/clus_ssl_arffs/fold{fold}_ssl_{scale}.arff"
        settings_file = f"./data/swdpf/clus_ssl_settings/fold{fold}_ssl_{scale}.s"
        generate_settings_from_arff(arff_file, settings_file)

