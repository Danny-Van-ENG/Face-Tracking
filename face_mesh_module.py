import cv2
import mediapipe as mp

class FaceMeshDetector:

    def __init__(self, static_mode = False,
                 max_faces = 1,
                 ref_landmarks = False,
                 min_detection_con = .5,
                 min_tracking_con = .5):

        self.static_mode = static_mode
        self.max_faces = max_faces
        self.ref_landmarks = ref_landmarks
        self.min_detection_con = min_detection_con
        self.min_tracking_con = min_tracking_con

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.drawing_spec_landmark = self.mp_drawing_styles.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        self.drawing_spec_connection = self.mp_drawing_styles.DrawingSpec(color=(220, 240, 239), thickness=1, circle_radius=1)

        self.mp_face_mesh = mp.solutions.face_mesh

        # object creation
        self.faceMesh = self.mp_face_mesh.FaceMesh(self.static_mode,
                                                   self.max_faces,
                                                   self.ref_landmarks,
                                                   self.min_detection_con,
                                                   self.min_tracking_con)

    def find_face_mesh(self, image, draw=True):
        self.imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imageRGB)
        faces = []

        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                # Landmark placement
                if draw:
                    self.mp_draw.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=self.drawing_spec_landmark,
                        connection_drawing_spec=self.drawing_spec_landmark)

                # Landmark positional data
                face = []
                for landmark_id, landmark in enumerate(face_landmarks.landmark):
                    # display landmark pixel cords
                    image_height, image_width, image_channel = image.shape
                    x, y = int(landmark.x * image_width), int(landmark.y * image_height) # add Z value here
                    face.append([x, y])
                faces.append(face)
        return image, faces
