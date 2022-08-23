# Video based sentiment analysis using a multimodal solution
This is a command based script for sentiment analysis it is based on 3 AI models the main model extracts 7 emotions from a detected face.
The other two models are there to assist with this job for a better accuracy. The first model is based on the eyes while the second one is based on the the mouth we combine these 3 models to get the detected emotion with a good accuracy.
To learn more about this please read [our paper](https://github.com/rykerdz/multimodal-emotion-recognition/blob/main/PFE%20COMPLET(1).pdf).

_This scripts uses the built in camera in your device to start a stream and gives the emotion of the detected faces_

## Installation
```
git clone https://github.com/rykerdz/multimodal-emotion-recognition
cd multimodal-emotion-recognition
pip install -r requirements.txt
```
## Usage
run ```python main.py```

## Requirements
- Python 3.9
- OpenCv
- dlib
- numpy
- onnx
- imutils
- PIL

## Contributors
- [AMOURA Youcef](https://github.com/rykerdz)
- [MOKHBAT Selma](https://github.com/SelmaMokhbat)

