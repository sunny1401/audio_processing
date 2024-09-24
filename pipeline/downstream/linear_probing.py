import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class SklearnModel:

    model_type={
        "multi-class" : LogisticRegression(
            multi_class="multinomial", solver='lbfgs', max_iter=20000, random_state=42),
        "binary": LogisticRegression(solver='lbfgs', max_iter=20000, random_state=42),
        "multi-label": OneVsRestClassifier(LogisticRegression(solver='lbfgs', max_iter=20000, random_state=42))
    }

    def __init__(self, train_file, test_file, model_type, val_file = None):
        base_train_data = np.load(train_file)
        base_test_data = np.load(test_file)
        self.x_train = base_train_data["features"]
        self.y_train = base_train_data["labels"]

        self.x_test = base_test_data["features"]
        self.y_test = base_test_data["labels"]

        if val_file: 
            base_val_data = np.load(val_file)
            self.x_val = base_val_data["features"]
            self.y_val = base_val_data["labels"]

        self.model = self.model_type[model_type]    
        
    def fit(self):
        self.model.fit(self.x_train, self.y_train)

    def get_val_preds(self):
        return self.model.predict(self.x_val)

    def get_test_preds(self):
        return self.model.predict(self.x_test)
    

def get_metrics(y_true, y_preds):

    f1 = f1_score(y_true, y_preds, average='weighted')
    acc = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_preds, average='weighted')

    print("F1 Score:", f1)
    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)