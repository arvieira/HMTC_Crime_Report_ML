import glob
import os

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
# from IPython.display import Image, display

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

from tqdm import tqdm
tqdm.pandas()


def load_model(path):
    model = XGBClassifier()
    model.load_model(path)
                
    return model

def read_encoder(path):
    encoder = LabelEncoder()
    encoder.classes_ = np.load(path, allow_pickle=True)
                                
    return encoder


model_db = {
    "N0_root": {
        "model": load_model("../../Training/Models/HierarchicalXGBoost/N0/root/root.model"),
        "encoder": read_encoder('../../Training/Models/HierarchicalXGBoost/N0/root/root.npy'),
    },
    "N1_Crimes Contra Pessoa": {
        "model": load_model("../../Training/Models/HierarchicalXGBoost/N1/Crimes Contra Pessoa/Crimes Contra Pessoa.model"),
        "encoder": read_encoder('../../Training/Models/HierarchicalXGBoost/N1/Crimes Contra Pessoa/Crimes Contra Pessoa.npy'),
    },
    "N1_Crimes Contra Propriedade": {
        "model": load_model("../../Training/Models/HierarchicalXGBoost/N1/Crimes Contra Propriedade/Crimes Contra Propriedade.model"),
        "encoder": read_encoder('../../Training/Models/HierarchicalXGBoost/N1/Crimes Contra Propriedade/Crimes Contra Propriedade.npy'),
    },
    "N1_Crimes de Trânsito ou Meio Ambiente": {
        "model": load_model("../../Training/Models/HierarchicalXGBoost/N1/Crimes de Trânsito ou Meio Ambiente/Crimes de Trânsito ou Meio Ambiente.model"),
        "encoder": read_encoder('../../Training/Models/HierarchicalXGBoost/N1/Crimes de Trânsito ou Meio Ambiente/Crimes de Trânsito ou Meio Ambiente.npy'),
    },
    "N1_Recuperação de Veículo ou Atos Administrativos ou Registro de Acontecimentos": {
        "model": load_model("../../Training/Models/HierarchicalXGBoost/N1/Recuperação de Veículo ou Atos Administrativos ou Registro de Acontecimentos/Recuperação de Veículo ou Atos Administrativos ou Registro de Acontecimentos.model"),
        "encoder": read_encoder('../../Training/Models/HierarchicalXGBoost/N1/Recuperação de Veículo ou Atos Administrativos ou Registro de Acontecimentos/Recuperação de Veículo ou Atos Administrativos ou Registro de Acontecimentos.npy'),
    },
    "N1_Relacionados a Drogas, Entorpecentes e Porte de Armas": {
        "model": load_model("../../Training/Models/HierarchicalXGBoost/N1/Relacionados a Drogas, Entorpecentes e Porte de Armas/Relacionados a Drogas, Entorpecentes e Porte de Armas.model"),
        "encoder": read_encoder('../../Training/Models/HierarchicalXGBoost/N1/Relacionados a Drogas, Entorpecentes e Porte de Armas/Relacionados a Drogas, Entorpecentes e Porte de Armas.npy'),
    },
    "N1_Resistência, Desacato ou Desobediência": {
        "model": load_model("../../Training/Models/HierarchicalXGBoost/N1/Resistência, Desacato ou Desobediência/Resistência, Desacato ou Desobediência.model"),
        "encoder": read_encoder('../../Training/Models/HierarchicalXGBoost/N1/Resistência, Desacato ou Desobediência/Resistência, Desacato ou Desobediência.npy'),
    },
    "N1_Violação ou Perturbação ou Dano ou Exercício Arbitrário": {
        "model": load_model("../../Training/Models/HierarchicalXGBoost/N1/Violação ou Perturbação ou Dano ou Exercício Arbitrário/Violação ou Perturbação ou Dano ou Exercício Arbitrário.model"),
        "encoder": read_encoder('../../Training/Models/HierarchicalXGBoost/N1/Violação ou Perturbação ou Dano ou Exercício Arbitrário/Violação ou Perturbação ou Dano ou Exercício Arbitrário.npy'),
    },
    "N2_Ameaça ou Injúria ou Perseguição ou Dano ou Extorsão": {
        "model": load_model("../../Training/Models/HierarchicalXGBoost/N2/Ameaça ou Injúria ou Perseguição ou Dano ou Extorsão/Ameaça ou Injúria ou Perseguição ou Dano ou Extorsão.model"),
        "encoder": read_encoder('../../Training/Models/HierarchicalXGBoost/N2/Ameaça ou Injúria ou Perseguição ou Dano ou Extorsão/Ameaça ou Injúria ou Perseguição ou Dano ou Extorsão.npy'),
    },
    "N2_Atos Administrativos": {
        "model": load_model("../../Training/Models/HierarchicalXGBoost/N2/Atos Administrativos/Atos Administrativos.model"),
        "encoder": read_encoder('../../Training/Models/HierarchicalXGBoost/N2/Atos Administrativos/Atos Administrativos.npy'),
    },
    "N2_Estelionato": {
        "model": load_model("../../Training/Models/HierarchicalXGBoost/N2/Estelionato/Estelionato.model"),
        "encoder": read_encoder('../../Training/Models/HierarchicalXGBoost/N2/Estelionato/Estelionato.npy'),
    },
    "N2_Estupro": {
        "model": load_model("../../Training/Models/HierarchicalXGBoost/N2/Estupro/Estupro.model"),
        "encoder": read_encoder('../../Training/Models/HierarchicalXGBoost/N2/Estupro/Estupro.npy'),
    },
    "N2_Furto": {
        "model": load_model("../../Training/Models/HierarchicalXGBoost/N2/Furto/Furto.model"),
        "encoder": read_encoder('../../Training/Models/HierarchicalXGBoost/N2/Furto/Furto.npy'),
    },
    "N2_Homicídio": {
        "model": load_model("../../Training/Models/HierarchicalXGBoost/N2/Homicídio/Homicídio.model"),
        "encoder": read_encoder('../../Training/Models/HierarchicalXGBoost/N2/Homicídio/Homicídio.npy'),
    },
    "N2_Lesão Corporal": {
        "model": load_model("../../Training/Models/HierarchicalXGBoost/N2/Lesão Corporal/Lesão Corporal.model"),
        "encoder": read_encoder('../../Training/Models/HierarchicalXGBoost/N2/Lesão Corporal/Lesão Corporal.npy'),
    },
    "N2_Recuperação de Veículo": {
        "model": load_model("../../Training/Models/HierarchicalXGBoost/N2/Recuperação de Veículo/Recuperação de Veículo.model"),
        "encoder": read_encoder('../../Training/Models/HierarchicalXGBoost/N2/Recuperação de Veículo/Recuperação de Veículo.npy'),
    },
    "N2_Registro de Um Acontecimento": {
        "model": load_model("../../Training/Models/HierarchicalXGBoost/N2/Registro de Um Acontecimento/Registro de Um Acontecimento.model"),
        "encoder": read_encoder('../../Training/Models/HierarchicalXGBoost/N2/Registro de Um Acontecimento/Registro de Um Acontecimento.npy'),
    },
    "N2_Roubo": {
        "model": load_model("../../Training/Models/HierarchicalXGBoost/N2/Roubo/Roubo.model"),
        "encoder": read_encoder('../../Training/Models/HierarchicalXGBoost/N2/Roubo/Roubo.npy'),
    },
}


# States
class OverallState(TypedDict):  
    predictions: pd.DataFrame

# Input state
class InputState(TypedDict):
    dataset_path: str

# Output state
class OutputState(TypedDict):
    # predictions: pd.DataFrame
    y_pred: list[np.int64]
    y_test: list[np.int64]
    proba: list[np.float32]


# First node function
def read_test_data(state: InputState) -> OverallState:
    print("NODE: Reading data.")

    data = pd.read_csv(state["dataset_path"])
                
    return {
        # Eu coloquei o nome de predictions pq as previsões serão feitas no próprio dataset, acrescentando as colunas
        "predictions": data,
    }


# Second node function
def predict_N0(state: OverallState) -> OverallState:
    print("NODE: Predicting N0.")

    predictions = state["predictions"]

    X = predictions.drop(columns=['N1', 'N2', 'N3'], axis=1)
    pred = model_db["N0_root"]["model"].predict(X)
    pred_proba = model_db["N0_root"]["model"].predict_proba(X)

    maximos = []
    for i in range(len(pred_proba)):
        maximos.append(pred_proba[i][pred[i]])
                                                                                                
    predictions["N1_pred"] = model_db["N0_root"]["encoder"].inverse_transform(pred)
    predictions["N1_pred_proba"] = maximos

    return {
        "predictions": predictions,
    }


# Third Node Aux function
def predict_row_n1(row):
    model_name = "N1_" + row["N1_pred"]
            
    X = row.drop(labels=['N1', 'N2', 'N3', 'N1_pred', 'N1_pred_proba'])
    X = X.values.reshape(1, -1)
                        
    pred = model_db[model_name]["model"].predict(X)
    pred_proba = max(model_db[model_name]["model"].predict_proba(X)[0])

    row["N2_pred"] = model_db[model_name]["encoder"].inverse_transform(pred)[0]
    row["N2_pred_proba"] = pred_proba

    return row

# Third node function
def predict_N1(state: OverallState) -> OverallState:
    print("NODE: Predicting N1.")

    predictions = state["predictions"].progress_apply(predict_row_n1, axis=1)

    return {
        "predictions": predictions
    }


# Fouth Node Aux function
def predict_row_n2(row):
    model_name = "N2_" + row["N2_pred"]
    if model_name in model_db.keys():    
        X = row.drop(labels=['N1', 'N2', 'N3', 'N1_pred', 'N1_pred_proba', 'N2_pred', 'N2_pred_proba'])
        X = X.values.reshape(1, -1)
                                        
        pred = model_db[model_name]["model"].predict(X)
        pred_proba = max(model_db[model_name]["model"].predict_proba(X)[0])
                                                            
        row["N3_pred"] = model_db[model_name]["encoder"].inverse_transform(pred)[0]
        row["N3_pred_proba"] = pred_proba
    else:
        row["N3_pred"] = row["N2_pred"]
        row["N3_pred_proba"] = 1

    return row

# Fourth node function
def predict_N2(state: OverallState) -> OutputState:
    print("NODE: Predicting N2.")

    predictions = state["predictions"].progress_apply(predict_row_n2, axis=1)

    return {
        "y_pred": predictions["N3_pred"],
        "y_test": predictions["N3"],
        "proba": predictions["N1_pred_proba"] * predictions["N2_pred_proba"] * predictions["N3_pred_proba"]
    }




# Graph
builder = StateGraph(OverallState, input=InputState, output=OutputState)

# Nodes
builder.add_node("Read and Split", read_test_data)
builder.add_node("Predict N0", predict_N0)
builder.add_node("Predict N1", predict_N1)
builder.add_node("Predict N2", predict_N2)

# Edges
builder.add_edge(START, "Read and Split")
builder.add_edge("Read and Split", "Predict N0")
builder.add_edge("Predict N0", "Predict N1")
builder.add_edge("Predict N1", "Predict N2")
builder.add_edge("Predict N2", END)

# Compile
graph = builder.compile()

# Evaluate
result = graph.invoke({"dataset_path": "../../../CE1/Datasets/13_NeverSeenANON.csv"})
report = classification_report(result["y_test"], result["y_pred"], digits=5, output_dict=True)
print(classification_report(result["y_test"], result["y_pred"], digits=5))
with open("./classification_report.txt", 'w') as f:
    print(classification_report(result["y_test"], result["y_pred"], digits=5), file=f)

accuracy_result = accuracy_score(result["y_test"], result["y_pred"])
print(f"Accuracy: {accuracy_result}")

cm = confusion_matrix(result["y_test"], result["y_pred"])
labels = list(report.keys())[:-3]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp = disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='vertical')
plt.grid(False)
plt.savefig(f"./confusion_matrix.png", bbox_inches="tight")
