import cv2
import csv
import os
from datetime import datetime

def mark_attendance(student_id):
    with open("attendance.csv", "a", newline="") as f:
        writer = csv.writer(f)
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([student_id, time, "Present"])

def recognize_student():
    if not os.path.exists("trainer.yml"):
        print("Train model first!")
        return

    face_classifier = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml"
    )

    model = cv2.face.LBPHFaceRecognizer_create()
    model.read("trainer.yml")

    marked = set()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            id, conf = model.predict(face)

            if conf < 60:
                if id not in marked:
                    mark_attendance(id)
                    marked.add(id)

                label = f"ID: {id}"
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()
