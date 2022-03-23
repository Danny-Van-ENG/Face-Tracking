import cv2
import mediapipe as mp
import time  # used to display framerate

mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=2)

drawing_spec_landmark = mp_drawing_styles.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
drawing_spec_connection = mp_drawing_styles.DrawingSpec(color=(220, 240, 239), thickness=1, circle_radius=1)

mp_face_mesh = mp.solutions.face_mesh
faceMesh = mp_face_mesh.FaceMesh(max_num_faces=1)  # object creation

# Webcam Input
cap = cv2.VideoCapture("0")
pTime = 0  # for framerate

while True:
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imageRGB)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Landmark placement
            mp_draw.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec_landmark,
                connection_drawing_spec=drawing_spec_connection)

            # Landmark positional data
            for id, landmark in enumerate(face_landmarks.landmark):
                # print(landmark) # landmark coords; these are normalized values 0 - 1

                # display landmark pixel cords
                image_height, image_width, image_channel = image.shape
                x, y = int(landmark.x * image_width), int(landmark.y * image_height) # add Z value here
                print(f"Landmark ID: {id} \nX: {x} \nY: {y} \n")

    # Framerate
    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(image, f'FPS: {int(fps)}', (20, 15),
                cv2.FONT_HERSHEY_PLAIN, 1,
                (0, 255, 0), 2)

    cv2.imshow("Image", image)
    cv2.waitKey(1)


