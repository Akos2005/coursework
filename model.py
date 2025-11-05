import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Masking
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix


from dataLoad import prepare_for_rnn, create_gesture_database

# RNN
data_dir = "combined"
csv_file = "gesture_database.csv"
gesture_df = create_gesture_database(data_dir, csv_file)

# Check
if gesture_df.empty:
    raise ValueError("Nincs adat a CSV-ben, a modell nem futtatható.")

# 3. Prepare  RNN
X_train, X_val, X_test, y_train, y_val, y_test, gesture_mapping = prepare_for_rnn(gesture_df, timesteps=110)

n_classes = len(gesture_mapping)
timesteps = X_train.shape[1]
features = X_train.shape[2]

# 4. 1-hot codes
y_train_cat = to_categorical(y_train, num_classes=n_classes)

# Val/Test
y_val_cat = to_categorical(y_val, num_classes=n_classes) if X_val.size > 0 else None
y_test_cat = to_categorical(y_test, num_classes=n_classes) if X_test.size > 0 else None

# --- Modell ---
model = Sequential([
    Masking(mask_value=0.0, input_shape=(timesteps, features)),
    SimpleRNN(128, return_sequences=True, activation="tanh"),
    Dropout(0.3),
    SimpleRNN(64, activation="tanh"),
    Dropout(0.3),
    Dense(n_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --- learning ---
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat) if X_val.size > 0 else None,
    epochs=50,
    batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)]
)

# --- Rating ---
if X_test.size > 0 and y_test_cat is not None:
    test_loss, test_acc = model.evaluate(X_test, y_test_cat)
    print(f"Test accuracy: {test_acc:.3f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
else:
    print("No test data")
    