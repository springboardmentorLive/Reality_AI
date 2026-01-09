import cv2
import os

def create_dataset(student_id):
    face_classifier = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml"
    )

    if not os.path.exists("dataset"):
        os.makedirs("dataset")

    cap = cv2.VideoCapture(0)
    img_id = 0

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            img_id += 1
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            file_name = f"dataset/{student_id}_{img_id}.jpg"
            cv2.imwrite(file_name, face)

            cv2.putText(face, str(img_id), (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Register Face", face)

        if cv2.waitKey(1) == 13 or img_id == 100:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Dataset creation complete")




