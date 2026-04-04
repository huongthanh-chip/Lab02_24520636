import numpy as np
import idx2numpy as idx2np
import matplotlib.pyplot as plt
import seaborn as sns
from logistic_regression import LogisticRegression

images = idx2np.convert_from_file('train-images.idx3-ubyte')
labels = idx2np.convert_from_file('train-labels.idx1-ubyte')
images_test = idx2np.convert_from_file('t10k-images.idx3-ubyte')
labels_test = idx2np.convert_from_file('t10k-labels.idx1-ubyte')

def filter_data(data, condition):
    images, labels = data
    new_images = images[condition]
    new_labels = labels[condition]
    return new_images, new_labels

train_images_0, train_label_0 = filter_data((images, labels), labels == 0)
train_images_1, train_label_1 = filter_data((images, labels), labels == 1)
test_images_0, test_label_0 = filter_data((images_test, labels_test), labels_test == 0)
test_images_1, test_label_1 = filter_data((images_test, labels_test), labels_test == 1)

train_images = np.concatenate((train_images_0, train_images_1), axis=0) / 255.0
train_labels = np.concatenate((train_label_0, train_label_1), axis=0)
test_images = np.concatenate((test_images_0, test_images_1), axis=0) / 255.0
test_labels = np.concatenate((test_label_0, test_label_1), axis=0)

#flatten images
N, _, _ = train_images.shape
train_images = train_images.reshape(N, -1) # (N, 28*28)
N, _, _ = test_images.shape
test_images = test_images.reshape(N, -1) # (N, 28*28)

model = LogisticRegression(
epoch = 50, 
lr = 0.01
)

indices = np.random.permutation(train_images.shape[0])
train_images = train_images[indices]
train_labels = train_labels[indices]   

model.fit(train_images, train_labels)
predicted_labels = model.predict(test_images)
metrics = model.evaluate(test_labels, predicted_labels)
for metric in metrics:
    print(f'{metric}: {metrics[metric]:.2f}')
model.plot_loss()