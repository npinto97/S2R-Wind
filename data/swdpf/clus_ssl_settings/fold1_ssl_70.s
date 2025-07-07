[General]
RandomSeed = 42

[Data]
File = C:/Users/Ningo/Desktop/S2R-Wind/data/swdpf/clus_ssl_arffs\fold1_ssl_70.arff
TestSet = C:/Users/Ningo/Desktop/S2R-Wind/data/swdpf/clus_ssl_arffs\fold1_test.arff
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

