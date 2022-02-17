import json
import os
import librosa
import math
import numpy as np

CURRENT_USER = "LOL"
DATASET_PATH = "dataset"
JSON_PATH = "dataset_output_" + str(CURRENT_USER) + ".json"
COUNT_PATH = "count_users.txt"

SAMPLE_RATE = 22050
DURATION = 239  # sec。最後都是一樣的規格長度
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(dataset_path, json_path, num_segments, n_mfcc=13, n_fft=2048, hop_length=512):

    # dictionary to store data
    TOTAL_CATEGORY = 0
    data = {
        "mapping": [],  # ["classical", "blues"],
        "mfcc": [],  # [[...], [...], [...]],
        "labels": []  # [0, 0, 1]
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure that we're not at root level
        if dirpath is not dataset_path:
            # save the semantic label
            dirpath_components = dirpath.split("\\") # dataset/classical -> ["dataset", "blues"] # 注意一下是'/'還是'\\'
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            TOTAL_CATEGORY = TOTAL_CATEGORY + 1
            print("\n Processing {}".format(semantic_label))

            # process files for a specific genre
            for f in filenames:
                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # process segments extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)

                    mfcc = mfcc.T
                    # print(mfcc.shape)

                    # store mfcc for segment if it has expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        if semantic_label == CURRENT_USER:
                            # data["labels"].append(i - 1) # i -1
                            data["labels"].append(1)
                        else:
                            data["labels"].append(0)
                        # data["labels"].append(semantic_label)
                        print("{}, segment: {}".format(file_path, s+1))

            print(mfcc.shape)

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

    print("Total category: {}".format(TOTAL_CATEGORY))
    with open(COUNT_PATH, 'w') as cp:
        cp.write("{}".format(TOTAL_CATEGORY))
        cp.close()


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, DURATION)