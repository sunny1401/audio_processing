import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier


class SklearnSegmentation:

    model_type={
            "multi-class" : LogisticRegression(
                multi_class="multinomial", solver='lbfgs', max_iter=20000, random_state=42),
            "binary": LogisticRegression(solver='lbfgs', max_iter=20000, random_state=42),
            "multi-label": OneVsRestClassifier(LogisticRegression(solver='lbfgs', max_iter=20000, random_state=42)),
            "_rf": RandomForestClassifier(n_estimators=100, random_state=42),
        }

    def __init__(self, train_file, test_file, model_type, val_file = None):
        base_train_data = np.load(train_file)
        base_test_data = np.load(test_file)
        self.x_train = base_train_data["features"]
        self.y_train = base_train_data["labels"]
        self.x_tr_processed, self.y_tr_processed = self.preprocess(self.x_train, self.y_train)

        self.x_test = base_test_data["features"]
        self.y_test = base_test_data["labels"]
        self.x_tst_processed, self.y_tst_processed = self.preprocess(self.x_test, self.y_test)

        if val_file: 
            base_val_data = np.load(val_file)
            self.x_val = base_val_data["features"]
            self.y_val = base_val_data["labels"]
            self.x_val_processed, self.y_val_processed = self.preprocess(self.x_val, self.y_val)

        self.model = self.model_type[model_type]    


    def mask_to_patch(self, mask, patch_size=16):
        """
        Convert a 224x224 mask to 14x14 patches using majority voting.
        """
        b, _, h, w = mask.shape
        patch_h, patch_w = h // patch_size, w // patch_size
        mask_patches = np.zeros((b, 1, patch_h, patch_w), dtype=int)
    
        for i in range(patch_h):
            for j in range(patch_w):
                patch = mask[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                mask_patches[:, :, i, j] = (np.sum(patch, axis=(2, 3)) > (patch_size * patch_size / 2)).astype(int)
    
        mask_patches = mask_patches.reshape(b, -1)  # Flatten to (batch_size, number_of_patches)
        return mask_patches

    def preprocess(self, X, Y):
        patches = X[:, 1:, :]
        x_processed = patches.reshape(-1, X.shape[-1])
        y_patches = self.mask_to_patch(Y) 
        y_processed = y_patches.reshape(-1)

        return x_processed, y_processed
        
    def fit(self):
        self.model.fit(self.x_tr_processed, self.y_tr_processed)

    def get_val_preds(self):
        return self.model.predict(self.x_val_processed)

    def get_test_preds(self):
        return self.model.predict(self.x_tst_processed)
    
    def patches_to_image(self, patches, image_size=224, patch_size=16):
    
        batch_size, patch_h, patch_w = patches.shape
        images = np.zeros((batch_size, 1, image_size, image_size), dtype=int)

        for b in range(batch_size):
            for i in range(patch_h):
                for j in range(patch_w):
                    images[b, 0, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patches[b, i, j]
        
        return images

    def plot_masks(self, images, original_mask, predicted_masks, rgb_index, index=110, alpha=0.5):
        _, axes = plt.subplots(1, 3, figsize=(18, 6))

        rgb_image = images[index][0][rgb_index].permute(1, 2, 0) 
        axes[0].imshow(rgb_image)
        axes[0].set_title("RGB Image")
        axes[0].axis('off')

        original_mask = np.squeeze(original_mask[index, 0]) 

        axes[1].imshow(rgb_image, alpha=1.0)
        axes[1].imshow(original_mask, cmap='jet', alpha=alpha)
        axes[1].set_title("Original Mask Overlay")
        axes[1].axis('off')

        predicted_mask = np.squeeze(predicted_masks[index, 0])
        axes[2].imshow(rgb_image, alpha=1.0) 
        axes[2].imshow(predicted_mask, cmap='jet', alpha=alpha)
        axes[2].set_title("Predicted Mask Overlay")
        axes[2].axis('off')

        plt.show()


def calculate_iou(pred, target):
    """
    Calculate Intersection over Union (IoU).
    """
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    iou = intersection / union if union != 0 else 0
    return iou

def calculate_dice(pred, target):
    """
    Calculate Dice Coefficient.
    """
    intersection = np.logical_and(pred, target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum()) if (pred.sum() + target.sum()) != 0 else 0
    return dice

def calculate_metrics(pred, target):
    """
    Calculate various metrics for segmentation.
    """
    pred = pred.flatten()
    target = target.flatten()

    iou = calculate_iou(pred, target)
    dice = calculate_dice(pred, target)
    precision = precision_score(target, pred, zero_division=1)
    recall = recall_score(target, pred, zero_division=1)
    f1 = f1_score(target, pred, zero_division=1)
    
    return iou, dice, precision, recall, f1
