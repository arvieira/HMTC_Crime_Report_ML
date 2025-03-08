import os
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .evaluator import Evaluator


def model_training(data, seed, tree, level, model_name, params):
    levels = tree.get_levels()
    
    if level != "N0":
        data = data[data[level] == model_name]

    X = data.drop(columns=levels[1:], axis=1)
    
    next_level = levels.index(level) + 1
    y = data[levels[next_level]]

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=seed
    )

    params = params[f"{level}_{model_name}"]

    model = XGBClassifier()

    print("Starting xgboost training.\nLet's work out and sweat!")
    grid = GridSearchCV(
            estimator=model,
            param_grid=params,
            scoring='accuracy',
            cv=StratifiedKFold(),
            verbose=3
    )
    grid.fit(X_train, y_train)
    
    print("That's all folks!")
    print(grid.best_estimator_)
    print(grid.best_params_) 

    xgboost = grid.best_estimator_

    dir_path = f"./Models/HierarchicalXGBoost/{level}/{model_name}"
    save_model(xgboost, f"{dir_path}/{model_name}.model", le, f"{dir_path}/{model_name}.npy")

    evaluator = Evaluator(xgboost, le, X_train, X_test, y_train, y_test)
    evaluator.evaluate(output_path=dir_path)
