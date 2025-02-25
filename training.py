import pickle
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.optimizers import Adam


data_dict = pickle.load(open('/Users/ishanshsharma/PycharmProjects/plz work /data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

data_dict['labels'] = encoded_labels

# Optionally save the updated pickle file (uncomment this to save)
# with open('/Users/ishanshsharma/PycharmProjects/plz work /data.pickle', 'wb') as f:
#     pickle.dump(data_dict, f)
print("1")

train_and_validation_data, test_data, train_and_validation_label, test_label = train_test_split(
    data, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
)
train_data, validation_data, train_label, validation_label = train_test_split(
    train_and_validation_data, train_and_validation_label, test_size=0.1, random_state=42, stratify=train_and_validation_label
)
print("3")

# Define the model
model = Sequential([
    layers.Input(shape=(data.shape[1],)),        # Input shape matches the number of coordinates (features)
    layers.Dense(128, activation='relu'),        # Hidden layer 1
    layers.Dropout(0.3),                         # Dropout for regularization
    layers.Dense(64, activation='relu'),         # Hidden layer 2
    layers.Dropout(0.3),                         # Dropout for regularization
    layers.Dense(32, activation='relu'),         # Hidden layer 3
    layers.Dense(len(np.unique(encoded_labels)), activation='softmax')  # Output layer (classes = unique labels)
])

print("ye bhi sahi hai")

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("dome")

# Train the model
history = model.fit(
    train_data, train_label,
    validation_data=(validation_data, validation_label),
    epochs=100,                                     # Adjust epochs based on performance
    batch_size=32,                                 # Batch size for training
    verbose=1                                      # Display training progress
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_data, test_label)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save the trained model
model.save('hand_gesture_ann.h5')

