import cv2 as cc
face_cascade = cc.CascadeClassifier(
    cc.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    print("Haar Cascade not loaded")
    exit()
cap = cc.VideoCapture(0)
if not cap.isOpened():
    print("Camera not accessible")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to get frame")
        break

    gray = cc.cvtColor(frame, cc.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,   
        minNeighbors=4,
        minSize=(80, 80)   
    )

    for (a, b, c, d) in faces:
        cc.rectangle(frame, (a, b), (a + c, b + d), (0, 255, 0), 2)
    cc.imshow("Face Detection", frame)
    if cc.waitKey(1) == 27:
        break

cap.release()
cc.destroyAllWindows()

