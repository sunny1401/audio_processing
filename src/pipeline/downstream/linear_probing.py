import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score


class SklearnModel:

    model_type = {
        "common": LogisticRegression(solver='lbfgs', max_iter=20000, random_state=42, n_jobs=-1),
        "multi-label": MultiOutputClassifier(LogisticRegression(solver='lbfgs', max_iter=20000, random_state=42, n_jobs=-1))
    }

    def __init__(self, train_file, test_file, model_type="common", val_file=None, use_pca=False, pca_components=50):
        # Load data
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

        # PCA option
        self.use_pca = use_pca
        if self.use_pca:
            self.pca = PCA(n_components=pca_components)
            self.x_train = self.pca.fit_transform(self.x_train)
            self.x_test = self.pca.transform(self.x_test)
            if val_file:
                self.x_val = self.pca.transform(self.x_val)

        # Initialize model
        self.model = self.model_type[model_type]    

    def fit(self):
        self.model.fit(self.x_train, self.y_train)

    def get_val_preds(self):
        return self.model.predict(self.x_val)

    def get_test_preds(self):
        return self.model.predict(self.x_test)
    

def get_metrics(y_true, y_preds, multi_label=False):
    if multi_label:
        metrics = f1_score(y_true, y_preds, average='micro')
        print("F1 Score (micro):", metrics)
    else:
        metrics = accuracy_score(y_true, y_preds)
        print("Accuracy (micro):", metrics)
    return metrics
