import uvicorn
import numpy as np
import librosa
from librosa import feature
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model

n_mfcc = 13
max_len = 300

model = load_model("LSTM_model.keras")

app = FastAPI()

def preprocess_audio(file_path: str):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T

    if mfcc.shape[0] < max_len:
        pad = np.zeros((max_len - mfcc.shape[0], n_mfcc))
        mfcc = np.vstack((mfcc, pad))
    else:
        mfcc = mfcc[:max_len, :]

    return np.expand_dims(mfcc.astype(np.float32), axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    X = preprocess_audio(file_path)

    pred = model.predict(X)
    gender = "female" if pred[0][0] > 0.5 else "male"

    return {"prediction": gender, "probability": float(pred[0][0])}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
