import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.metrics import accuracy_score

#importing dataset with pandas.

train = pd.read_csv('dataset/sign-language-mnist/sign-mnist-train.csv')
test = pd.read_csv('dataset/sign-language-mnist/sign-mnist-test.csv')

#peeking into the dataset

train.head()

#shape of the train dataset

train.shape

#labels of the train dataset

labels = train['label'].values

#finding the unique values / classes in the train dataset labels

unique_val = np.array(labels)
np.unique(unique_val)

#visualizing the distribution of train dataset labels

plt.figure(figsize = (18,8))
sns.countplot(x =labels)

#drop the label column in the train dataset

train.drop('label', axis = 1, inplace = True)

#reshaping the train dataset images into 28x28 pixels and flatten the images
images = train.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])

#Instancing a LabelBinarizer and transform the labels into the binary form

label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)

#showing the first resized image

plt.imshow(images[0].reshape(28,28))

#splitting the dataset to test and train with 0.2 as test size

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, random_state = 101)

#defining batch_size, num_classes, and epochs. And reshaping the train and test data into 28x28 pixels with a single grayscale channel


batch_size = 64
num_classes = 24
epochs = 50

x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
plt.imshow(x_train[0].reshape(28,28))

#Defining a sequential model and adding extra layers to the model and compile the model and train

model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28,28,1) ))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20))

model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)
model.save("ASL_TO_TEXT_MP4_V1.h5")

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('ASL to Text Conversion Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('ASL to Text Conversion Model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

#Preparing test data for evaluation. Extracting labels from test data. Reshaping the test data images from flatten array to 2D array with 28x28 pixels. And converting categorical labels to binary vectors.

test_labels = test['label']

test.drop('label', axis = 1, inplace = True)

test_images = test.values
test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])
test_images = np.array([i.flatten() for i in test_images])

test_labels = label_binrizer.fit_transform(test_labels)

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

test_images.shape

#Making predictions with test data. And getting the accuracy score

y_pred = model.predict(test_images)
accuracy_score(test_labels, y_pred.round())

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(test_labels, axis=1)

# Compute the confusion matrix
conf_mat = confusion_matrix(y_true_classes, y_pred_classes)

# Plot the confusion matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=label_binrizer.classes_, yticklabels=label_binrizer.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('ASL to Text Conversion Model Confusion Matrix')
plt.show()


import pandas as pd

# Generate the classification report as a dictionary
report = classification_report(y_true_classes, y_pred_classes, target_names=label_binrizer.classes_, output_dict=True)

# Create a DataFrame from the report dictionary
metrics_df = pd.DataFrame(report).transpose()

# Select and rename columns for clarity
metrics_df = metrics_df[['precision', 'recall', 'f1-score', 'support']]
metrics_df.columns = ['Precision', 'Recall', 'F1-Score', 'Support']

# Styling the DataFrame
styled_metrics_df = metrics_df.style.set_table_styles([
    {'selector': 'th', 'props': [('border', '1px solid black'), ('background-color', '#01B5CB')]},
    {'selector': 'td', 'props': [('border', '1px solid black')]}
]).set_properties(**{'text-align': 'center'})

# Display the styled DataFrame
styled_metrics_df

import matplotlib.pyplot as plt
import pandas as pd

# Generate the classification report as a dictionary
report = classification_report(y_true_classes, y_pred_classes, target_names=label_binrizer.classes_, output_dict=True)

# Create a DataFrame from the report dictionary
metrics_df = pd.DataFrame(report).transpose()

# Select and rename columns for clarity
metrics_df = metrics_df[['precision', 'recall', 'f1-score', 'support']]
metrics_df.columns = ['Precision', 'Recall', 'F1-Score', 'Support']

# Plotting the table
fig, ax = plt.subplots(figsize=(10, 6))  # set size frame
ax.axis('off')  # no axes

# Create the table
table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, rowLabels=metrics_df.index, loc='center', cellLoc='center')

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # Increase size for better readability

# Save the table as an image
plt.savefig('classification_report_table.png', bbox_inches='tight', dpi=300)
plt.show()