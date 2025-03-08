import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from yellowbrick.classifier import ROCAUC
from yellowbrick.features import RadViz
from sklearn.model_selection import train_test_split


SEED = 12345


def plot_roc_multiclass(model, label_encoder, output_path, X, y):
    viewer = ROCAUC(model, classes=label_encoder.classes_, size=(1250, 900))

    X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=SEED
    )

    custom_viewer = viewer.ax
    custom_viewer.figure.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3
    )
    custom_viewer.figure.show()


def evaluate(alg, real, predicted, pred_proba, classes):
    # Preparando labels
    columns = [classe + " Pred" for classe in classes]
    indexes = [classe + " Real" for classe in classes]

    # Matriz de confusão
    print(f"\nAvaliando o {alg} com matriz de confusão:")
    cm = confusion_matrix(real, predicted)
    print(metrics.classification_report(real, predicted, digits=5))
    with open("./classification_report.txt", 'w') as f:
        print(metrics.classification_report(real, predicted, digits=5), file=f)

    # AUC
    print(f'-> Acurácia: {accuracy_score(real, predicted)}')
    print(f"-> Valor AUC: {metrics.roc_auc_score(real, pred_proba, multi_class='ovr')}\n")

    # Exibindo a matriz de confusão para colocar no artigo
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(40,40))
    disp = disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='vertical')
    plt.grid(False)
    plt.savefig(f"./confusion_matrix.png", bbox_inches="tight")
    plt.clf()


def evaluate_naive_model(model_path, label_encoder_path, dataset_path):
    model = XGBClassifier()
    model.load_model(model_path)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(label_encoder_path, allow_pickle=True)

    test_data = pd.read_csv(dataset_path)

    df_X_test = test_data.drop(columns=['N1', 'N2', 'N3'], axis=1)
    df_y_test = label_encoder.fit_transform(test_data['N3'])

    df_y_pred = model.predict(df_X_test)
    pred_proba = model.predict_proba(df_X_test) # [::, 1]
    evaluate("XGBoost", df_y_test, df_y_pred, pred_proba, label_encoder.classes_)

    plot_roc_multiclass(model, label_encoder, ".", df_X_test, df_y_test)



evaluate_naive_model(
    model_path="../../Training/Models/NaiveXGBoost/NaiveXGBoost.model",
    label_encoder_path="../../Training/Models/NaiveXGBoost/NaiveXGBoost.npy",
    dataset_path="../../../CE1/Datasets/13_NeverSeenANON.csv"
)
