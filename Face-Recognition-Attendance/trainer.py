import cv2
import os
import numpy as np

def train_model():
    data_path = "dataset"
    faces = []
    ids = []

    for file in os.listdir(data_path):
        path = os.path.join(data_path, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        id = int(file.split("_")[0])

        faces.append(img)
        ids.append(id)

    ids = np.array(ids)

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(faces, ids)
    model.save("trainer.yml")

    print("Training complete")


