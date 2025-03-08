from matplotlib import pyplot as plt

from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from yellowbrick.classifier import ROCAUC
from yellowbrick.features import RadViz



class Evaluator:
    def __init__(self, model, label_encoder, X_train, X_test, y_train, y_test):
        self.model = model
        self.label_encoder = label_encoder
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


    def plot_roc_binary(self, prediction_proba, roc_auc, output_path):
        fpr, tpr, thresh = roc_curve(self.y_test, prediction_proba)
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(fpr, tpr, label=f"XGBoost - AUC: {roc_auc:.3f}")

        ax.set_title('ROC Curve')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc=0)
        
        plt.savefig(f"{output_path}/roc_curve.png", bbox_inches="tight") 


    def plot_roc_multiclass(self, output_path):
        viewer = ROCAUC(self.model, classes=self.label_encoder.classes_, size=(1250, 900))
        viewer.fit(self.X_train, self.y_train)
        viewer.score(self.X_test, self.y_test)
        viewer.show(outpath=f"{output_path}/roc_curve.png")


    def analyze(self, y_pred, prediction_proba, output_path):
        # Pegando as classes
        classes = self.label_encoder.classes_

        # Preparando labels
        columns = [classe + " Pred" for classe in classes]
        indexes = [classe + " Real" for classe in classes]
        
        # Métricas
        report = classification_report(self.y_test, y_pred, digits=5)
        accuracy = accuracy_score(self.y_test, y_pred)
        if len(classes) == 2:
            roc_auc = roc_auc_score(self.y_test, prediction_proba)
        else:
            roc_auc = roc_auc_score(self.y_test, prediction_proba, multi_class='ovr')
        with open(f"{output_path}/metrics.txt", "w") as file:
            file.write(report)
            file.write(f"\nAUC: {roc_auc}")

        # Matriz de Confusão
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.label_encoder.classes_)
        disp = disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='vertical')
        plt.grid(False)
        plt.savefig(f"{output_path}/confusion_matrix.png", bbox_inches="tight") 
        plt.clf()

        if len(classes) == 2:
            self.plot_roc_binary(prediction_proba, roc_auc, output_path)
        else:
            self.plot_roc_multiclass(output_path)


    def evaluate(self, output_path=None):
        if output_path:
            y_pred = self.model.predict(self.X_test)
            if len(self.label_encoder.classes_) == 2:
                prediction_proba = self.model.predict_proba(self.X_test)[::, 1]
            else:
                prediction_proba = self.model.predict_proba(self.X_test)

            self.analyze(y_pred, prediction_proba, output_path)

