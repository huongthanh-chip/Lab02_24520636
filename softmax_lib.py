import numpy as np
import idx2numpy as idx2np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

from softmax_regression import SoftmaxRegression

print("Loading data...")
images = idx2np.convert_from_file('train-images.idx3-ubyte')
labels = idx2np.convert_from_file('train-labels.idx1-ubyte')
images_test = idx2np.convert_from_file('t10k-images.idx3-ubyte')
labels_test = idx2np.convert_from_file('t10k-labels.idx1-ubyte')

print("Data loaded")
train_images = images / 255.0
train_labels = labels
test_images = images_test / 255.0
test_labels = labels_test

#flatten images
N, _, _ = train_images.shape
train_images = train_images.reshape(N, -1) # (N, 28*28)
N, _, _ = test_images.shape
test_images = test_images.reshape(N, -1) # (N, 28*28)

X_train = train_images
y_train = train_labels
X_test = test_images
y_test = test_labels

indices = np.random.permutation(X_train.shape[0])
X_train = X_train[indices][:1000]  
y_train = y_train[indices][:1000]

model = LogisticRegression (
    multi_class='multinomial',
    max_iter=100,
    solver='lbfgs',
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
