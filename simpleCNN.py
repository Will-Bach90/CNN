# William Bach
# CSS 486 Autumn 2023

# This code is adapted from Hardik Desmukh's medium article here:
# https://towardsdatascience.com/medical-x-ray-%EF%B8%8F-image-classification-using-convolutional-neural-network-9a6d33b1c2a
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
# Used when saving and loading the model
# from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# # Define paths
train_path = 'pneumonia-dataset/train'
test_path = 'pneumonia-dataset/test'
valid_path = 'pneumonia-dataset/val'

# Define standard parameters
batch_size = 16
img_height = 500
img_width = 500
#
#


# Function to create data generators for training, validation, and testing
def create_data_generators(batch_size, img_height, img_width):
    # Data augmentation and preprocessing for training data
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Rescale the pixel values 
        shear_range=0.2,  # Randomly apply shearing transformations
        zoom_range=0.2,   # Randomly zoom inside pictures
        horizontal_flip=True  # Randomly flip images horizontally
    )

    # Only rescaling for validation and test data
    test_datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values

    # Generator that will read images found in the directory, 
    # and generate batches of augmented/normalized data
    train_generator = train_datagen.flow_from_directory(
        train_path,  # Path to the training data
        target_size=(img_height, img_width),  # Resize images to specified dimensions
        color_mode='grayscale',  # Images will be converted to grayscale
        class_mode='binary',  # Binary labels (two classes)
        batch_size=batch_size  # Size of the batches of data
    )

    # Similar generator for validation data (without augmentation)
    validation_generator = test_datagen.flow_from_directory(
        valid_path,  # Path to the validation data
        target_size=(img_height, img_width),
        color_mode='grayscale',
        class_mode='binary',
        batch_size=batch_size
    )

    # Similar generator for test data (without augmentation)
    test_generator = test_datagen.flow_from_directory(
        test_path,  # Path to the test data
        target_size=(img_height, img_width),
        color_mode='grayscale',
        shuffle=False,  # Do not shuffle the order of the images
        class_mode='binary',
        batch_size=batch_size
    )

    # Return the data generators
    return train_generator, validation_generator, test_generator


# # Function to create the CNN model
def create_cnn_model(img_height, img_width):
    model = Sequential()  # Initialize a Sequential model in Keras

    # Add a 2D Convolutional layer with 32 filters, kernel size of 3x3, and ReLU activation.
    # The input_shape specifies the shape of the input images.
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))

    # Add a Max Pooling layer with a 2x2 window to reduce spatial dimensions.
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output of the previous layer to a 1D vector to feed into a dense layer.
    model.add(Flatten())

    # Add a Dense (fully connected) layer with a sigmoid activation function for binary classification output.
    model.add(Dense(activation='sigmoid', units=1))

    # Compile the model: 'adam' optimizer and 'binary_crossentropy' loss.
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
#


# # Function to train the model
def train_model(model, train_generator, validation_generator):
    # Set up an early stopping callback to stop training when the validation loss does not improve.
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)

    # Set up a callback to reduce the learning rate when the validation loss plateaus.
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.3, min_lr=0.000001)

    # Combine the callbacks into a list.
    callbacks_list = [early_stop, reduce_lr]

    # Compute class weights to handle class imbalance in training data.
    weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
    class_weights = dict(zip(np.unique(train_generator.classes), weights))

    # Fit the model to the data. This will train the model on the training data with specified epochs, 
    # and validate it on the validation data. Class weights and callbacks are also used.
    history = model.fit(
        train_generator,
        epochs=25,  # Number of epochs to train for
        validation_data=validation_generator,  # Data for validation
        class_weight=class_weights,  # Handling class imbalances
        callbacks=callbacks_list  # Early stopping and learning rate reduction
    )

    return history


# # Initialize data generators
train_gen, valid_gen, test_gen = create_data_generators(batch_size, img_height, img_width)
#
# # Create and train the model
cnn_model_simplified = create_cnn_model(img_height, img_width)
history = train_model(cnn_model_simplified, train_gen, valid_gen)

# # Used when saving and later reloading the model
# cnn_model_simplified.save('simplified_model.h5')
# cnn_model_simplified = load_model('my_model.h5')

# Evaluate the model on the test data and print the testing accuracy
test_accu = cnn_model_simplified.evaluate(test_gen)
print('Testing accuracy:', test_accu[1] * 100, '%')

# Generate predictions for the test data
predictions = cnn_model_simplified.predict(test_gen, verbose=1)

# Convert probabilities to binary predictions (0 or 1) based on a threshold of 0.5
predictions[predictions <= 0.5] = 0
predictions[predictions > 0.5] = 1


# # Generate classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
cm = pd.DataFrame(data=confusion_matrix(test_gen.classes, predictions, labels=[0, 1]),
                  index=["Actual Normal", "Actual Pneumonia"],
                columns = ["Predicted Normal", "Predicted Pneumonia"])
print(cm)
plt.figure(figsize=(8, 6)) 
sns.heatmap(cm, cmap="crest")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print(classification_report(test_gen.classes, predicted_classes, target_names=["Normal", "Pneumonia"]))