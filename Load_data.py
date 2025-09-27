import os
import json
import numpy as np
import librosa
from librosa import feature

# --- Ustawienia ---
path = r'C:/Users/Paweł/Desktop/ML/Datasets kaggle/Audio MNIST/'
path_to_y = os.path.join(path, 'audioMNIST_meta.txt')
n_mfcc = 13        # liczba MFCC
max_len = 300      # liczba kroków czasowych
folders1 = [f'0{i}' for i in range(1,10)]
folders2 = [f'{j}{i}' for i in range(1,10) for j in range(1,6)]
folders3 = ['60']
folders = folders1 + folders2 + folders3

# --- Wczytanie metadanych ---
with open(path_to_y, 'r') as f:
    y_dict = json.load(f)

# --- Wczytywanie danych i MFCC ---
X_list = []
y_list = []

for folder in folders:
    folder_path = os.path.join(path, folder)
    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith('.wav'):
            continue
        file_path = os.path.join(folder_path, file_name)
        y_audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc)
        mfcc = mfcc.T

        # Padding lub skrócenie
        if mfcc.shape[0] < max_len:
            pad = np.zeros((max_len - mfcc.shape[0], n_mfcc), dtype=np.float32)
            mfcc = np.vstack((mfcc, pad))
        else:
            mfcc = mfcc[:max_len, :]

        X_list.append(mfcc.astype(np.float32))

        # Tylko płeć
        if folder in y_dict:
            gender_str = y_dict[folder].get("gender", "male")  # domyślnie male
            gender = 1 if str(gender_str).lower() == "female" else 0
        else:
            gender = 0  # domyślnie male, jeśli brak metadanych
        y_list.append(gender)

np.save('X_list.npy', X_list)
np.save('y_list.npy', y_list)
