import pandas as pd

from packages.tree_mount import HierarchyTree
from packages.xgboost_training import model_training
from packages.params import params


SEED = 12345

print("Mounting hierarchy tree")
tree = HierarchyTree("../../CE1/Datasets/04_DomainHierarchy.csv")

print("Starting data reading")
df = pd.read_csv("../../CE1/Datasets/10_RawTrainANON.csv")

# Grid search para o modelo hierarquico
model_training(df, SEED, tree, 'N0', 'root', params)

model_training(df, SEED, tree, 'N1', 'Crimes Contra Pessoa', params)
model_training(df, SEED, tree, 'N1', 'Crimes Contra Propriedade', params)
model_training(df, SEED, tree, 'N1', 'Crimes de Trânsito ou Meio Ambiente', params)
model_training(df, SEED, tree, 'N1', 'Recuperação de Veículo ou Atos Administrativos ou Registro de Acontecimentos', params)
model_training(df, SEED, tree, 'N1', 'Relacionados a Drogas, Entorpecentes e Porte de Armas', params)
model_training(df, SEED, tree, 'N1', 'Resistência, Desacato ou Desobediência', params)
model_training(df, SEED, tree, 'N1', 'Violação ou Perturbação ou Dano ou Exercício Arbitrário', params)

model_training(df, SEED, tree, 'N2', 'Ameaça ou Injúria ou Perseguição ou Dano ou Extorsão', params)
model_training(df, SEED, tree, 'N2', 'Atos Administrativos', params)
model_training(df, SEED, tree, 'N2', 'Estelionato', params)
model_training(df, SEED, tree, 'N2', 'Estupro', params)
model_training(df, SEED, tree, 'N2', 'Furto', params)
model_training(df, SEED, tree, 'N2', 'Homicídio', params)
model_training(df, SEED, tree, 'N2', 'Lesão Corporal', params)
model_training(df, SEED, tree, 'N2', 'Recuperação de Veículo', params)
model_training(df, SEED, tree, 'N2', 'Registro de Um Acontecimento', params)
model_training(df, SEED, tree, 'N2', 'Roubo', params)

# Se for utilizar a base preprocessada, precisa realizar o procedimento de grid search do zero para cada um dos modelos e trocar a base
