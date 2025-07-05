import os

scales = [10, 20, 70]
folds = range(8)
TARGET_NAME = "patv_target_1"

def generate_settings_from_arff(arff_path, settings_path, target_name=TARGET_NAME):
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
TestSet = None
PruneSet = None

[Attributes]
Target = {target_index}
Weights = Normalize

[Model]
MinimalWeight = 2.0

[Tree]
Heuristic = VarianceReduction
FTest = 1.0
ConvertToRules = No

[Ensemble]
EnsembleMethod = RForest
Iterations = 30

[SemiSupervised]
SemiSupervisedMethod = SelfTraining
StoppingCriteria = NoneAdded
Iterations = 1
UnlabeledCriteria = KMostConfident
K = 5
ConfidenceMeasure = Variance
UseWeights = No
Normalization = NoNormalization
Aggregation = Average
OOBErrorCalculation = LabeledOnly
PercentageLabeled = {scale}

[Output]
ShowModels = Default
TrainErrors = No
TestErrors = Yes
AllFoldModels = No
AllFoldErrors = No
AllFoldDatasets = No
UnknownFrequency = No
BranchFrequency = No
ShowInfo = Count
PrintModelAndExamples = No
WriteModelFile = No
WritePredictions = Test
GzipOutput = No

"""

    with open(settings_path, "w") as f:
        f.write(settings_content)

    print(f"Settings written: {settings_path}")


for fold in folds:
    for scale in scales:
        arff_file = f"C:/Users/Ningo/Desktop/S2R-Wind/data/swdpf/clus_ssl_arffs/fold{fold}_ssl_{scale}.arff"
        settings_file = f"C:/Users/Ningo/Desktop/S2R-Wind/data/swdpf/clus_ssl_settings/fold{fold}_ssl_{scale}.s"
        generate_settings_from_arff(arff_file, settings_file)
