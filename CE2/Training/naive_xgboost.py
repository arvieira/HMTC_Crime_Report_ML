import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from packages.evaluator import Evaluator


SEED = 12345


if __name__ == "__main__":
    print("Reading database")
    df = pd.read_csv("../../CE1/Datasets/10_RawTrainANON.csv")

    # Se for ler a base de dados preprocessada, trocar a linha anterior pela abaixo
    # df = pd.read_csv("../../Datasets/10_PreprocTrainANON.csv")

    X = df.drop(columns=['N1', 'N2', 'N3'], axis=1)
    y = df['N3']

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=SEED
    )


    print("Starting training")
    model = XGBClassifier()
    params = {
        'booster': ['gbtree'],
        'objective': ['multi:softmax'],
        'num_class': [len(le.classes_)],
        'eval_metric': ['mlogloss'],
        'learning_rate': [0.1],
        'n_estimators': [100],
        'max_depth': [9],
        'min_child_weight': [8],
        'gamma': [0.015],
        'subsample': [0.5],
        'colsample_bytree': [0.2],
        'reg_alpha': [0.00005],
        'seed': [SEED],
        'device': ['cuda:0']
    }

    # Se for usar a base preprocessada, utilizar a matriz abaixo
    # params = {
    #     'booster': ['gbtree'],
    #     'objective': ['multi:softmax'],
    #     'num_class': [len(le.classes_)],
    #     'eval_metric': ['mlogloss'],
    #     'learning_rate': [0.1],
    #     'n_estimators': [100],
    #     'max_depth': [6],
    #     'min_child_weight': [10],
    #     'gamma': [0.06],
    #     'subsample': [0.5],
    #     'colsample_bytree': [0.5],
    #     'reg_alpha': [0.0001],
    #     'seed': [SEED],
    #     'device': ['cuda:1']
    # }

    grid = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring='accuracy',
        cv=StratifiedKFold(),
        verbose=3
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print(grid.best_estimator_)
    print(grid.best_params_)

    print("Saving model")
    dir_path = "./Models/NaiveXGBoost"
    best_model.save_model(f"{dir_path}/NaiveXGBoost.model")
    np.save(f"{dir_path}/NaiveXGBoost.npy", le.classes_)

    # Se for utilizar a base preprocesada, usar os nomes e caminhos abaixo
    # dir_path = "./Models/NaiveXGBoostPreproc"
    # best_model.save_model(f"{dir_path}/NaiveXGBoostPreproc.model")
    # np.save(f"{dir_path}/NaiveXGBoostPreproc.npy", le.classes_)

    print("Evaluating model")
    evaluator = Evaluator(best_model, le, X_train, X_test, y_train, y_test)
    evaluator.evaluate(output_path=dir_path)
