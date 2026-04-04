import numpy as np
import idx2numpy as idx2np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

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

print("Fitting model...")
model = LogisticRegression(
    max_iter=50,
    solver='saga',
    verbose=1
)

model.fit(X_train, y_train)
print("Predicting...")
predicted_labels = model.predict(X_test)
precision = precision_score(y_test, predicted_labels, average='macro')
recall = recall_score(y_test, predicted_labels, average='macro')
f1 = f1_score(y_test, predicted_labels, average='macro')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')