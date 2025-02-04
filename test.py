import tensorflow as tf
from LGDAAN_Net import LGDAAN_Net
from data_loader import load_data

def test(model, target_time_data, target_spectral_data):
    predictions = []
    for step in range(0, len(target_time_data), 32):
        target_time_batch = target_time_data[step:step + 32]
        target_spectral_batch = target_spectral_data[step:step + 32]

        emotion_preds, _, _, _ = model(target_time_batch, target_spectral_batch, target_time_batch, target_spectral_batch)
        predictions.append(emotion_preds)

    print("Predictions:", predictions)

model = LGDAAN_Net()
model.load_weights("LGDAAN_Net_weights")
source_time_data, source_spectral_data, target_time_data, target_spectral_data, source_labels, target_labels = load_data(
        './preprocess/DEAP_temp_sour', './preprocess/DEAP_spec_sour', './preprocess/DEAP_temp_targ', './preprocess/DEAP_spec_targ')

test(model, target_time_data, target_spectral_data)
