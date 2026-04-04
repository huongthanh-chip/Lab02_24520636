import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

class LogisticRegression:
    def __init__(self, epoch: int, lr: float):
        self.epoch = epoch
        self.lr = lr
        self.w = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, x1: np.ndarray, y1: np.ndarray):
        N, d = x1.shape
        self.w = np.zeros((d, 1), dtype=np.float64)
        y1 = y1.reshape(-1, 1)
        self.loss_history = []
        for e in tqdm(range(self.epoch)):
            z = x1 @ self.w
            y_pred = self.sigmoid(z)
            delta_y = y_pred - y1  # (N, 1)
            gradient = (1 / N) * x1.T @ delta_y  # (d, 1)
            self.w -= self.lr * gradient
            loss = self.loss_fn(y1, y_pred)
            self.loss_history.append(loss)
        self.epoch_loss = self.loss_history[-1] if self.loss_history else None

    def loss_fn(self, y1: np.ndarray, y_pred: np.ndarray) -> float: 
        l = (1 - y1) * np.log(1 - y_pred + 1e-8) + y1 * np.log(y_pred + 1e-8)
        return -l.mean()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.w 
        return self.sigmoid(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

    def evaluate(self, y, y_pred) -> dict: 
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def plot_loss(self):
        epochs = np.arange(1, len(self.loss_history) + 1)
        plt.plot(epochs, self.loss_history, marker='o', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.grid(True)
        plt.savefig('logistic_loss.png')
        plt.show()