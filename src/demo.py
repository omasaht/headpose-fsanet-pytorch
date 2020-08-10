from argparse import ArgumentParser
import numpy as np
import cv2
import onnxruntime
import sys
from pathlib import Path
#local imports
from face_detector import FaceDetector
from utils import draw_axis

root_path = str(Path(__file__).absolute().parent.parent)

def _main(cap_src):

    cap = cv2.VideoCapture(cap_src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_d = FaceDetector()

    sess = onnxruntime.InferenceSession(f'{root_path}/pretrained/fsanet-1x1-iter-688590.onnx')

    sess2 = onnxruntime.InferenceSession(f'{root_path}/pretrained/fsanet-var-iter-688590.onnx')

    print('Processing frames, press q to exit application...')
    while True:
        ret,frame = cap.read()
        if(not ret):
            print('Could not capture a valid frame from video source, check your cam/video value...')
            break
        #get face bounding boxes from frame
        face_bb = face_d.get(frame)
        for (x1,y1,x2,y2) in face_bb:
            face_roi = frame[y1:y2+1,x1:x2+1]

            #preprocess headpose model input
            face_roi = cv2.resize(face_roi,(64,64))
            face_roi = face_roi.transpose((2,0,1))
            face_roi = np.expand_dims(face_roi,axis=0)
            face_roi = (face_roi-127.5)/128
            face_roi = face_roi.astype(np.float32)

            #get headpose
            res1 = sess.run(["output"], {"input": face_roi})[0]
            res2 = sess2.run(["output"], {"input": face_roi})[0]

            yaw,pitch,roll = np.mean(np.vstack((res1,res2)),axis=0)

            draw_axis(frame,yaw,pitch,roll,tdx=(x2-x1)//2+x1,tdy=(y2-y1)//2+y1,size=50)

            #draw face bb
            # cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.imshow('Frame',frame)

        key = cv2.waitKey(1)&0xFF
        if(key == ord('q')):
            break




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--video", type=str, default=None,
                        help="Path of video to process i.e. /path/to/vid.mp4")
    parser.add_argument("--cam", type=int, default=None,
                        help="Specify camera index i.e. 0,1,2...")
    args = parser.parse_args()
    cap_src = args.cam if args.cam is not None else args.video
    if(cap_src is None):
        print('Camera or video not specified as argument, selecting default camera node (0) as input...')
        cap_src = 0
    _main(cap_src)
