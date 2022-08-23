import numpy as np
import cv2
import os
from imutils import face_utils
from torchvision import transforms
from PIL import Image
import numpy as np
import dlib
import cv2
import torch


# Defining directories
base_dir = os.path.dirname(__file__)+"/"
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))


def features_extraction(my_image, startX, startY, endX, endY, predictor):
    ''' this function takes as input image along with coordinates
        of the detected face (x, y, h, w) and draws rectangle around
        the mouth and eyes. the function also return the detected eyes
        and mouth as numpy array '''

    # Defining a black image to capture errors
    defaultPicture = np.zeros((60, 150, 1))

    #--- Mouth detection ---#
    # Converting to grayscale
    gray = cv2.cvtColor(my_image, cv2.COLOR_BGR2GRAY)
    dlibRect = dlib.rectangle(startX, startY, endX, endY)
    shape = predictor(gray, dlibRect)
    


    x1=shape.part(17).x
    x2=shape.part(26).x
    y1=shape.part(19).y
    y2=shape.part(41).y
    grayEyes = gray[y1-10:y2+10,x1-10:x2+10]
    #cv2.imwrite("results/test.jpg", cv2.resize(grayEyes, (224, 224)))

    shape = face_utils.shape_to_np(shape)

    draw_border(my_image, (x1-10, y1-10), (x2+10, y2+10),
                        (173, 216, 230), 2, 5, 10)
    
    cv2.putText(my_image, 'DETECTED EYES', (x1-10, y1-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (173, 216, 230), 1)

    eyes = process_img(grayEyes, 1)
    




    # loop over the face parts individually
    for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():

        if(name == 'mouth'):
            # extract the ROI of the face region as a separate image
            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))

            # Rectangle
            draw_border(my_image, (x-10, y-10), (x+w+10, y+h+10),
                        (173, 216, 230), 2, 5, 10)
            cv2.putText(my_image, 'DETECTED MOUTH', (x-10, y-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (173, 216, 230), 1)

            roi = gray[y-10:y + h+10, x-10:x + w+10]

            # if mouth was detected pass it to image pre-process
            mouth = defaultPicture
            if(np.sum(roi) != 0):
                # final variable to pass to the model
                mouth = process_img(roi, 1)

    #--- Eye pairs detection ---#
    grayFace = gray[startY:startY + endY, startX:startX + endX]
    # eyePairs = eyePair_cascade.detectMultiScale(grayFace)

    # eyes = defaultPicture
    # for (ex, ey, ew, eh) in eyePairs:  # eye_pair points
    #     grayEyes = grayFace[ey-20: ey+eh, ex-20:ex + ew]

    #     # rectangle around eyes
    #     draw_border(my_image, (startX+ex-15, startY+ey-15),
    #                 (startX+ex+ew+15, startY+ey+eh+15), (173, 216, 230), 2, 5, 10)

    #     cv2.putText(my_image, 'DETECTED EYES', (startX+ex-15, startY+ey-20),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (173, 216, 230), 1)

    #     eyes = process_img(grayEyes, 1)
    #     if(np.sum(eyes) != 0):
    #         is_eyes = True

    # processing face
    face = process_img(grayFace, 0)

    return face, mouth, eyes


def get_emotion(face_session, mouth_session, eyes_session, face, mouth, eyes, frame, startX, startY):
    ''' Gets face, eyes and mouth emotions and converts it to one emotion'''

    # Cheking for detected eyes and mouth
    is_mouth = True
    if(np.sum(mouth) == 0):
        is_mouth = False
    
    is_face = True
    if(np.sum(face) == 0):
        is_face = False
    
    
    is_eyes = True
    if(np.sum(eyes) == 0):
        is_eyes = False

    # Getting emotions
    emotion = ""

    # Initializing emotions dicts
    face_emotions_dict = {0: "angry",
                          1: "disgust",
                          2: "fear",
                          3: "happy",
                          4: "neutral",
                          5: "sad",
                          6: "surprise"}

    mouth_emotions_dict = {0: "angry",
                           1: "disgust",
                           2: "happy",
                           3: "sad",
                           4: "surprise",
                           5: "uncertain"}



    # if(is_eyes and is_mouth):
    # mouth, eyes and face
    if(is_face):
        #face
        face_inputs = {face_session.get_inputs()[0].name: face}
        face_outs = face_session.run(None, face_inputs)

        res = face_outs[0]
        ps = torch.exp(torch.from_numpy(res))

        # getting top_p and top classes
        top_p, top_class = ps.topk(1)
        # converting to list
        top_p = top_p.detach().numpy().tolist()[0]
        top_class = top_class.detach().numpy().tolist()[0]

        face_emotion = face_emotions_dict[top_class[0]]

        #mouth
        if (is_mouth):
            mouth_inputs = {mouth_session.get_inputs()[0].name: mouth}
            mouth_outs = mouth_session.run(None, mouth_inputs)

            res = mouth_outs[0]
            ps = torch.exp(torch.from_numpy(res))
            # getting top_p and top classes
            top_p, top_class = ps.topk(2)
            # converting to list
            top_p = top_p.detach().numpy().tolist()[0]
            top_class = top_class.detach().numpy().tolist()[0]

            mouth_emotion = [mouth_emotions_dict[top_class[i]] for i in list(range(2))]

            emotion = calculate_emotions(face_emotion, mouth_emotion, eyes_session, eyes)

            #test
            #print("face emotion: ", face_emotion, "Mouth emotion: ", mouth_emotion, "Emotion: ", emotion)
            #print(top_p)
            if(is_eyes):
                eyes_emotions_dict = {0: "angry",
                          1: "sad",
                          2: "surprise",
                          3: "uncertain"}
                # eyes
                eyes_inputs = {eyes_session.get_inputs()[0].name: eyes}
                eyes_outs = eyes_session.run(None, eyes_inputs)

                res = eyes_outs[0]
                ps = torch.exp(torch.from_numpy(res))

                # getting top_p and top classes
                top_p, top_class = ps.topk(2)
                # converting to list
                top_p = top_p.detach().numpy().tolist()[0]
                top_class = top_class.detach().numpy().tolist()[0]

                eyes_emotion = [eyes_emotions_dict[top_class[i]] for i in list(range(2))] 
                print("face emotion: ", face_emotion, "Mouth emotion: ", mouth_emotion, "eyes emotion:", eyes_emotion, "Emotion: ", emotion)
        
        if(is_eyes and not is_mouth):
            emotion = calculate_emotions(face_emotion, "", eyes_session, eyes)
        
        if(not is_eyes and not is_mouth):
            emotion = face_emotion
            
    else:
        emotion="Undefined"
    
    cv2.putText(frame, emotion, (startX, startY-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (225, 249, 126), 3)
            
    

    


def calculate_emotions(face, mouth, eyes_session, eyes):
    if(mouth==""):
        eyes_emo = use_eyes(eyes, eyes_session)
        if(face=="neutral"):
            if("uncertain" in eyes_emo):
                return "neutral"
            if("sad" in eyes_emo):
                return "sad"
            if("surprise" in eyes_emo):
                return "surprise"
            else:
                return "neutral"
        
        if(face=="sad"):
            return "sad"
        if(face=="disgust"):
            return "disgust"
        if(face=="surprise"):
            if("surprise" in eyes_emo):
                return "surprise"
            else:
                return "neutral"
        if(face=="angry"):
            return "angry"
        if(face=="happy"):
            return "happy"

        if(face=="fear"):
            return "fear"
            

    if(face=="neutral"):
        #if(mouth[0]=="sad"):
            #return "sad"
        # use eyes
        eyes_emo = use_eyes(eyes, eyes_session)
        if("sad" == eyes_emo[0]):
            if("sad" in mouth):
                return "sad"
        if("surprise"==eyes[0]):
            return "fear"
        if("uncertain" in mouth):
            return "neutral"
        
        else:
            return "neutral"



    if(face=="happy"):
        # if("disgust" == mouth[0]):
        #     return "disgust"
        # if(mouth[0]=="angry"):
        #     return "angry"
        if("sad" in mouth or "uncertain" in mouth):
            print("here")
            return "happy"

        # if("disgust" in mouth or "angry" in mouth or "happy" in mouth):
        #     return "disgust"
        
        
        
        else:
            return "happy"

    if(face=="disgust"):
        # if(mouth[0]=="sad" and (mouth[1]=="angry" or mouth[1]=="uncertain") or mouth[0]=="angry"):
        #     return "angry"
        # else:
            return "disgust"
    
    if(face=="angry"):
        # if(mouth[0]=="angry"):
        #     return "angry"
        if("happy" == mouth[0] and mouth[1]!="angry"):
            return "disgust"
        else:
            return "angry"

    if(face=="surprise"):
        # eyes_emo = use_eyes(eyes, eyes_session)
        # if("surprise" not in eyes_emo):
        #     return "neutral"

        if("surprise" in mouth):
            return "surprise"
        if("sad" or "uncertain" or "disgust" in mouth):
            return "fear"
        else:
            return "angry"

    if(face=="sad"):
        return "sad"

    if(face=="fear"):
        return "fear"
            
            
            


    if(type==1):
        # Face and mouth emotion
        pass
    if(type==2):
        # Face and eyes emotion
        
        pass


def use_eyes(eyes, eyes_session):

    eyes_emotions_dict = {0: "angry",
                          1: "sad",
                          2: "surprise",
                          3: "uncertain"}
    # eyes
    try:
        eyes_inputs = {eyes_session.get_inputs()[0].name: eyes}
        eyes_outs = eyes_session.run(None, eyes_inputs)
        res = eyes_outs[0]
        ps = torch.exp(torch.from_numpy(res))

        # getting top_p and top classes
        top_p, top_class = ps.topk(2)
        # converting to list
        top_p = top_p.detach().numpy().tolist()[0]
        top_class = top_class.detach().numpy().tolist()[0]

        eyes_emotion = [eyes_emotions_dict[top_class[i]] for i in list(range(2))] 

        return eyes_emotion
    except:
        print("No eyes")
        return ["", ""]


    
    
    


def process_img(img, type):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array. Type: 0->Face, 1->Eyes, mouth
    '''

    if(np.sum(img) == 0):
        # img_pil = Image.fromarray(img).convert('RGB')
        # img = img.astype(np.float32)
        return img

    # this works too
    img_pil = Image.fromarray(img).convert('RGB')
    transform = transforms.Compose([transforms.Resize(161),
                                    transforms.CenterCrop(160),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) if (type == 0) else transforms.Compose([transforms.Resize(225),
                                                                                                                                                 transforms.CenterCrop(
                                                                                                                                                224),
                                                                                                                                                transforms.ToTensor(),
                                                                                                                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    final_img = transform(img_pil)
    final_img = final_img.to(dtype=torch.float32)
    final_img.unsqueeze_(0)
    final_img = to_numpy(final_img)
    final_img = final_img.astype(np.float32)

    return final_img


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
