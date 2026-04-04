import numpy as np
import idx2numpy as idx2np
import matplotlib.pyplot as plt
import seaborn as sns
from logistic_regression import LogisticRegression
from softmax_regression import SoftmaxRegression

images = idx2np.convert_from_file('train-images.idx3-ubyte')
labels = idx2np.convert_from_file('train-labels.idx1-ubyte')
images_test = idx2np.convert_from_file('t10k-images.idx3-ubyte')
labels_test = idx2np.convert_from_file('t10k-labels.idx1-ubyte')

print('Images shape:', images.shape)
print('Labels shape:', labels.shape)

def convert_to_onehot_vector(labels: np.ndarray):
    N = labels.shape[0]
    total_classes = labels.max() + 1
    oh_labels = np.zeros((N, total_classes), dtype=np.float64)
    oh_labels[np.arange(N), labels] = 1
    return oh_labels


train_images = images / 255.0
train_labels = labels
test_images = images_test / 255.0
test_labels = labels_test
encoded_train_labels = convert_to_onehot_vector(train_labels)
encoded_test_labels = convert_to_onehot_vector(test_labels)

#flatten images
N, _, _ = train_images.shape
train_images = train_images.reshape(N, -1) # (N, 28*28)
N, _, _ = test_images.shape
test_images = test_images.reshape(N, -1) # (N, 28*28)

model = SoftmaxRegression(
epoch = 500, 
lr = 0.1
)

indices = np.random.permutation(train_images.shape[0])
train_images = train_images[indices]
encoded_train_labels = encoded_train_labels[indices]

model.fit(train_images, encoded_train_labels)
predicted_labels = model.predict(test_images)
metrics = model.evaluate(test_labels, predicted_labels)
for metric in metrics:
    print(f'{metric}: {metrics[metric]:.2f}')
model.plot_loss()
