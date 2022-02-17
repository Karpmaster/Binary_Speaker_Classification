import os
import librosa
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf


# FILE_PATH = '../Audio/Jazz3.wav'
FILE_PATH = 'Wayne_test.wav'
MODEL_PATH = 'trained_model_LOL'
SAMPLE_RATE = 22050
DURATION = 1  # sec。最後都是一樣的規格長度
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
accuracy_threshold = 0.8


def preprocess(FILE_PATH,  num_segments, n_mfcc=13, n_fft=2048, hop_length=512):

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    # expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # load file
    file_path = os.path.join(FILE_PATH)
    signal, sr = librosa.load(file_path, sr= SAMPLE_RATE)

    # extracting segments mfcc and storing data
    for s in range(num_segments):
        start_sample = num_samples_per_segment * s
        finish_sample = start_sample + num_samples_per_segment

        mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                    sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                    hop_length=hop_length)
        mfcc = mfcc.T

    print("mfcc shape: {}".format(mfcc.shape))

    mfcc = mfcc[..., np.newaxis]
    return mfcc


def check_accuracy(predictions):
    max_acc = None
    max_index = None
    # print(predictions[0])
    for i, (acc) in enumerate(predictions[0]):
        # print(max_acc)
        # print(i)
        if acc > accuracy_threshold:
            max_acc = acc
            max_index = i
            # print(max_acc)
            # print(max_index)
    return max_acc, max_index


def predict(model, file_mfcc):
    # 3D array -> 4D array
    file_mfcc = file_mfcc[np.newaxis, ...]
    print("file_mfcc shape: {}".format(file_mfcc.shape))
    # predictions = [[0.1, 0.2, 0.3, ...]]
    predictions = model.predict(file_mfcc)
    # extract index with max value
    print(predictions)
    # checked_accuracy, checked_index = check_accuracy(predictions)
    # predicted_index = np.argmax(predictions, axis=1)  # [3]
    # print("Predicted index: {}".format(predicted_index))
    # if checked_index is not None:
    #     print("Checked: {} ,{}".format(checked_accuracy, checked_index))
    # else:
    #     print("The Audio input doesn't belong to any category!")


if __name__ == "__main__":
    # extra file to predict but need to preprocess
    file_mfcc = preprocess(FILE_PATH, 1)
    # load model
    model = keras.models.load_model(MODEL_PATH)
    # make prediction on a sample
    predict(model, file_mfcc)

