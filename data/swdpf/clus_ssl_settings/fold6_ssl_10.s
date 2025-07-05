[General]
RandomSeed = 42

[Data]
File = C:/Users/Ningo/Desktop/S2R-Wind/data/swdpf/clus_ssl_arffs/fold6_ssl_10.arff
TestSet = None
PruneSet = None

[Attributes]
Target = 20
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
PercentageLabeled = 10

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

