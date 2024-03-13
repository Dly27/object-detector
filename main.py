import cv2
import numpy as np

# Load the pre-trained deep learning model for face detection
model_path = 'deploy.prototxt'
weights_path = 'mobilenet_iter_73000.caffemodel'
net = cv2.dnn.readNetFromCaffe(model_path, weights_path)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (1200, 600))
    frame = cv2.flip(frame, 1)

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.85:
            # Get the class label
            class_id = int(detections[0, 0, i, 1])
            class_label = CLASSES[class_id]

            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, w, h) = box.astype(int)

            cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)
            cv2.putText(frame, class_label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
