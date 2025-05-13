from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import os

# Parameters
path = 'myData' # Path to data directory
testRatio = 0.2
valRatio = 0.2
imageDimensions = (32, 32, 3)

# Importing Data
images = []     # List of Images
classNo = []    # Id of all the corresponding Images

# Get the list of subdirectories (classes) within 'myData'
myList = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

noOfClasses = len(myList)
print(f"Total Classes Detected: {noOfClasses}")

for x in range(0, noOfClasses):
    myPicList = os.listdir(path + "/" + str(myList[x]))
    print(f"Loading Class {myList[x]}: {len(myPicList)} images")
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(myList[x]) + "/" + y)
        # Check if the image was loaded successfully
        if curImg is not None:
            curImg = cv2.resize(curImg, (imageDimensions[0], imageDimensions[1]))
            images.append(curImg)
            classNo.append(x)
        else:
            print(f"Could not load image: {path}/{myList[x]}/{y}") 

# Converting Images to Numpy Array
images = np.array(images)
classNo = np.array(classNo)

print(f"Total Images: {len(images)}")
print(f"Shape of Images array: {images.shape}")

# Spliting data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    images, classNo, test_size=testRatio, random_state=42)
X_train, X_validation, y_train, y_validation = train_test_split(
    X_train, y_train, test_size=valRatio, random_state=42)

print(f"Training: {X_train.shape[0]} images")
print(f"Validation: {X_validation.shape[0]} images")
print(f"Testing: {X_test.shape[0]} images")

# Preprocessing
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255.0
    return img

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

# Reshape and Transform Images
X_train = X_train.reshape(
    X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(
    X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

# Data Augmentation
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

# One-Hot Encoding
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)

def myModel():  # Creating model
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNodes = 500

    model = Sequential()
    model.add(Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0],
                                                             imageDimensions[1], 1), activation='relu'))
    model.add(Conv2D(noOfFilters, sizeOfFilter1, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu'))
    model.add(Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())

# Model Training
batch_size = 32
epochs = 10

history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train)//batch_size,
                    epochs=epochs,
                    validation_data=(X_validation, y_validation),
                    verbose=1)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Score evaluation
score = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Score: {score[0]}")
print(f"Test Accuracy: {score[1]}")

# Saving model
model.save("model_trained.h5")
print("Model saved successfully!")