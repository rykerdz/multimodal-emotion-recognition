import os


import cv2
import numpy as np
import onnxruntime
import dlib
from torch import device
import utils

# Define paths
base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + '/model_data/deploy.prototxt')
caffemodel_path = os.path.join(base_dir + '/model_data/weights.caffemodel')

print("#Loading models...")
# Read the face detection model
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# loading the dlib predictor
#predictor = dlib.shape_predictor('./haar-cascade/shape_predictor_68_face_landmarks.dat')


device = "CPU_FP32"
# preparing the models
options = onnxruntime.SessionOptions()
options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC 

#options.optimized_model_filepath = "outpu\optimized_model1.onnx>"
#options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
face_session = onnxruntime.InferenceSession("models/face_emotions.onnx", options, providers=['OpenVINOExecutionProvider'], provider_options=[{'device_type': device}])
#face_session.set_providers(['OpenVINOExecutionProvider'], [{'device_type': device}])

print("--------------[ Loaded face model ]--------------")
options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC 

#options.optimized_model_filepath = "outpu\optimized_model2.onnx>"
eyes_session = onnxruntime.InferenceSession("models/eyes_emotions_final_low_uncertain2.onnx", options, providers=['OpenVINOExecutionProvider'], provider_options=[{'device_type': device}])

print("--------------[ Loaded eyes model ]--------------")
options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC 

#options.optimized_model_filepath = "outpu\optimized_model3.onnx>"
mouth_session = onnxruntime.InferenceSession("models/mouth_emotions_trained80.onnx", options, providers=['OpenVINOExecutionProvider'], provider_options=[{'device_type': device}])
print("--------------[ Loaded mouth model ]--------------")
print("\n--------------[ All models are loaded! Starting web-cam ]--------------")

def main():
    
    cam = cv2.VideoCapture(0)
    while True:
        check, frame = cam.read()
        if(check):
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(
                frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

            model.setInput(blob)
            detections = model.forward()

            # Create frame around face
            for i in range(0, detections.shape[2]):
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                confidence = detections[0, 0, i, 2]

                # If confidence > 0.5, show box around face
                if (confidence > 0.5):
                    utils.draw_border(frame, (startX, startY), (endX, endY), (225,249,126), 3, 25, 25)
                    
                    face, mouth, eyes = utils.features_extraction(frame, startX, startY, endX, endY, predictor)


                    # Emotion detection
                    utils.get_emotion(face_session, mouth_session, eyes_session, face, mouth, eyes, frame, startX, startY)
                    #cv2.putText(frame, emotion, (startX, startY-10),
                                #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (225,249,126), 1)
                    #utils.returnMouthAndEyes(frame, startX, startY, endX, endY)
                        

        cv2.imshow('video', frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
