import cv2
import pickle as pkl
import numpy as np
import sys
import os
sys.path.append('../')
from utils import measureDistance, measureXYdistance


class CameraMovementEstimator():
    def __init__(self,frame):

        self.minDistance = 5
        self.lkParams = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        )


        firstFrameGrayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        maskFeatures = np.zeros_like(firstFrameGrayScale)
        maskFeatures[:,0:20] = 1
        maskFeatures[:,900:1050] = 1
        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = maskFeatures
        )

    def adjustPositionsToTracks(self,tracks,cameraMovementPerFrame):
        for object, objectTracks in tracks.items():
            for frameNum, track in enumerate(objectTracks):
                for trackID, trackInfo in track.items():
                    position = trackInfo['position']
                    cameraMovement = cameraMovementPerFrame[frameNum]
                    positionAdjusted = (position[0] - cameraMovement[0], position[1] - cameraMovement[1])
                    tracks[object][frameNum][trackID]['positionAdjusted'] = positionAdjusted



    def getCameraMovement(self, frames,readFromStubs = False, stubPath = None):

        if readFromStubs and stubPath is not None and os.path.exists(stubPath):
            print(f"Carregando movimento da camera de {stubPath}")
            with open(stubPath,'rb') as f:
                return pkl.load(f)
        
        print(f"Calculando movimento da camera")
        cameraMovement = [[0,0]]*len(frames)

        oldGray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        oldFeatures = cv2.goodFeaturesToTrack(oldGray,**self.features)

        for frameNum in range(1,len(frames)):
            frameGrey = cv2.cvtColor(frames[frameNum], cv2.COLOR_BGR2GRAY)
            newFeatures, _,_ = cv2.calcOpticalFlowPyrLK(oldGray,frameGrey,oldFeatures,None,**self.lkParams)

            maxDistance = 0
            cameraMovement_x , cameraMovement_y = 0,0

            for i, (new,old) in enumerate(zip(newFeatures,oldFeatures)):
                newFeaturesPoint = new.ravel()
                oldFeaturesPoint = old.ravel()

                distance = measureDistance(newFeaturesPoint,oldFeaturesPoint)

                if distance > maxDistance:
                    maxDistance = distance
                    cameraMovement_x, cameraMovement_y = measureXYdistance(oldFeaturesPoint, newFeaturesPoint)
            if maxDistance < self.minDistance:
                cameraMovement[frameNum] = [cameraMovement_x, cameraMovement_y]
                oldFeatures = cv2.goodFeaturesToTrack(frameGrey,**self.features)

            oldGray = frameGrey.copy()
        
        if stubPath is not None:
            print(f"Salvando movimento da camera em {stubPath}")
            with open(stubPath,'wb') as f:
                pkl.dump(cameraMovement,f)
        return cameraMovement
    
    def drawCameraMovement(self,frames, camera_movement_per_frame):
        output_frames=[]

        for frame_num, frame in enumerate(frames):
            frame= frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha =0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            output_frames.append(frame) 
        print(f"Movimento da camera desenhado!")
        return output_frames