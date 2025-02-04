import tensorflow as tf
from tensorflow.keras import optimizers, losses

def train_step(model, source_time, source_spectral, target_time, target_spectral, source_labels, optimizer):
    with tf.GradientTape() as tape:
        emotion_preds, local_time_preds, local_spectral_preds, global_preds = model(
            source_time, source_spectral, target_time, target_spectral)

        # 损失计算
        local_time_loss = losses.SparseCategoricalCrossentropy()(source_labels, local_time_preds)
        local_spectral_loss = losses.SparseCategoricalCrossentropy()(source_labels, local_spectral_preds)
        global_loss = losses.SparseCategoricalCrossentropy()(source_labels, global_preds)
        classification_loss = losses.SparseCategoricalCrossentropy()(source_labels, emotion_preds)

        total_loss = local_time_loss + local_spectral_loss + global_loss + classification_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss

def train(model, source_time_data, source_spectral_data, target_time_data, target_spectral_data, source_labels, epochs, batch_size, learning_rate):
    optimizer = optimizers.Adam(learning_rate)
    total_loss = 0

    for epoch in range(epochs):
        for step in range(0, len(source_time_data), batch_size):
            source_time_batch = source_time_data[step:step + batch_size]
            source_spectral_batch = source_spectral_data[step:step + batch_size]
            target_time_batch = target_time_data[step:step + batch_size]
            target_spectral_batch = target_spectral_data[step:step + batch_size]
            source_labels_batch = source_labels[step:step + batch_size]

            loss = train_step(model, source_time_batch, source_spectral_batch, target_time_batch, target_spectral_batch, source_labels_batch, optimizer)
            total_loss += loss

        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss.numpy()}')

    model.save_weights("LGDAAN_Net_weights")
    print("Model weights saved.")
