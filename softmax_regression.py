import numpy as np
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

class SoftmaxRegression:
    def __init__(self, epoch: int, lr: float):
        self.epoch = epoch
        self.lr = lr
        self.w = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, x1: np.ndarray, y1: np.ndarray):
        N, d = x1.shape
        _, k = y1.shape
        self.w = np.zeros((d, k), dtype=np.float64)
        self.loss_history = []
        for e in tqdm.tqdm(range(self.epoch), desc='Training'):
            z = x1 @ self.w
            y_pred = self.softmax(z)
            delta_y = y_pred - y1  # (N, k)
            gradient = (1 / N) * x1.T @ delta_y  # (d, k)
            self.w -= self.lr * gradient
            self.epoch_loss = self.loss_fn(y1, y_pred)
            self.loss_history.append(self.epoch_loss)
        if self.loss_history:
            self.epoch_loss = self.loss_history[-1]
        else:
            self.epoch_loss = None

    def loss_fn(self, y1: np.ndarray, y_pred: np.ndarray) -> float: 
        return - (y1 * np.log(y_pred + 1e-8)).sum(axis = 1).mean() # (N, 3)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.w 
        return self.softmax(z)

    def softmax(self, z: np.ndarray):
        denum = np.exp(z).sum(axis=1, keepdims=True)
        return np.exp(z) / denum
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def evaluate(self, y, y_pred) -> dict: 
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
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
        plt.savefig('softmax_loss.png')
        plt.show()
    