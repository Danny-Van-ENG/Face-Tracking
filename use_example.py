import cv2
import face_mesh_module as fmm
import time

cap = cv2.VideoCapture(0)
pTime = 0  # for framerate
detector = fmm.FaceMeshDetector()

while cap.isOpened():
    success, image = cap.read()
    image, faces = detector.find_face_mesh(image, draw=True)

    # Display Landmark Data
    if len(faces) != 0:
        print(faces[0])

    # Framerate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(image, f'FPS: {int(fps)}', (20, 15),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    # flip image for selfie-view
    cv2.imshow("Image", cv2.flip(image, 1))
    cv2.waitKey(1)
