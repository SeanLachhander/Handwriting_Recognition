import matplotlib.pyplot as plt

# Import the library/dataset from scikit-learn package
from sklearn.datasets import load_digits
digits = load_digits()

# Analyze sample image
import pylab as pl
pl.gray()
pl.matshow(digits.images[0])
pl.show()

# Analyze image pixels (Matrix representing grayscale pixels)
print(digits.images[0])

images_and_labels = list(zip(digits.images, digits.target))
plt.figure(figsize=(5,5))
for index, (image, label) in enumerate(images_and_labels[:15]):
    plt.subplot(3,5,index+1)
    plt.axis('off')
    plt.title('%i' % label)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')

import random
from sklearn import ensemble

n_samples = len(digits.images)
x = digits.images.reshape((n_samples, -1))
y = digits.target

# Create random indices
import math
sample_index = random.sample(range(len(x)), math.floor(len(x)/5))

valid_index = [i for i in range(len(x)) if i not in sample_index]

# Sample and validation images

sample_images = [x[i] for i in sample_index]
valid_images = [x[i] for i in valid_index]

# Sample and validation targets

sample_target = [y[i] for i in sample_index]
valid_target = [y[i] for i in valid_index]

# Use the Random Tree Classifier

classifier = ensemble.RandomForestClassifier()

# Fit model with sample data

classifier.fit(sample_images, sample_target)

# Attempt to predict validation data
score=classifier.score(valid_images, valid_target)
print("Random Tree Classifier Score: {}".format(str(score)))

# Use Nearest Neighbors Approach

from sklearn.neighbors import KNeighborsClassifier
nearest_neighbors_classifier = KNeighborsClassifier()

# Fit model with sample data

nearest_neighbors_classifier.fit(sample_images, sample_target)

# Attempt to predict validation data

kNNScore=nearest_neighbors_classifier.score(valid_images, valid_target)

print("Nearest Neighbors Classifier Score: {}".format(str(kNNScore)))
