import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.callbacks import EarlyStopping

X_list = np.load('X_list.npy')
y_list = np.load('y_list.npy')


X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

n_mfcc = 13
max_len = 300

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.5, random_state=42)

model = Sequential([
    Masking(mask_value=0., input_shape=(max_len, n_mfcc)),
    LSTM(50),
    Dense(20, activation='relu')
])

model.add(Dense(1, activation='sigmoid', name='gender'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


early_stop = EarlyStopping(
    monitor= 'val_loss',
    patience= 3,
    restore_best_weights=True

)

model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks= [early_stop]
)

loss, accuracy = model.evaluate(X_test, y_test, batch_size=32, verbose=1)

model.save("LSTM_model.keras")
