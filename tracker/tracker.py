from ultralytics import YOLO
import supervision as sv
import pickle as pkl
import os
import sys
import cv2
import pandas as pd
import numpy as np
sys.path.append("../")
from utils import getCenterOfBBox, getBboxWidth, getFootPosition

class Tracker:
    def __init__(self,modelPath):
        self.model = YOLO(modelPath)
        self.tracker = sv.ByteTrack()

    def addPositionToTrack(self,tracks):
        for object, objectTracks in tracks.items():
            for frameNum, track in enumerate(objectTracks):
                for trackID, trackInfo in track.items():
                    bbox = trackInfo["bbox"]
                    if object == "ball":
                        position = getCenterOfBBox(bbox)
                    else:
                        position = getFootPosition(bbox)
                    tracks[object][frameNum][trackID]["position"] = position

    def interpolateBall(self,ballPositions):
        ballPositions = [x.get(1,{}).get("bbox",[]) for x in ballPositions]
        dfBallPositions = pd.DataFrame(ballPositions,columns=["x1","y1","x2","y2"])

        #interpolar valores faltantes
        dfBallPositions = dfBallPositions.interpolate()
        dfBallPositions = dfBallPositions.bfill()

        ballPositions = [{1 : {"bbox":x}}for x in dfBallPositions.to_numpy().tolist()]
        print(f"Bola interpolada!")
        return ballPositions

    def detectFrames(self,frames):
        batchSize = 20
        detections = []
        for i in range(0,len(frames),batchSize):
            detections += (self.model.predict(frames[i:i+batchSize],conf=0.1))
        return detections

    def getObjectTracks(self,frames,readFromStub = False, stubPath=None):

        if readFromStub and stubPath is not None and os.path.exists(stubPath):
            print(f"Carregando tracks de {stubPath}")
            with open(stubPath,"rb") as f:
                return pkl.load(f)
            
        print(f"Sem trakcs, Detectando objetos nos frames!")
        detections = self.detectFrames(frames)

        tracks = {
            "players":[],  
            "ball":[],
            "referees":[]
        }

        for frameNum, detection in enumerate(detections):
            classNames = detection.names
            classNamesInv = {value:key for key,value in classNames.items()}

            detectionSupervision = sv.Detections.from_ultralytics(detection)

            # Converter goleiro para jogador
            for objectIndex, classID in enumerate(detectionSupervision.class_id):
                if classNames[classID] == "goalkeeper":
                    detectionSupervision.class_id[objectIndex] = classNamesInv["player"]

            # Track objects
            detectionWithTracks = self.tracker.update_with_detections(detectionSupervision)
            tracks["players"].append({})
            tracks["ball"].append({})
            tracks["referees"].append({})

            for frameDetection in detectionWithTracks:
                bbox = frameDetection[0].tolist()
                clsID = frameDetection[3]
                trackID = frameDetection[4]
                
                if clsID == classNamesInv["player"]:
                    tracks["players"][frameNum][trackID] = {"bbox":bbox}
                if clsID == classNamesInv["referee"]:
                    tracks["referees"][frameNum][trackID] = {"bbox":bbox}
            
            for frameDetection in detectionSupervision:
                bbox = frameDetection[0].tolist()
                clsID = frameDetection[3]

                if clsID == classNamesInv["ball"]:
                    tracks["ball"][frameNum][1] = {"bbox":bbox}
        if stubPath is not None:   
            print(f"Salvando tracks em {stubPath}")
            with open(stubPath,"wb") as f:
                pkl.dump(tracks,f) 

        return tracks
    
    def drawTriangle(self,frame,bbox,color):
        y = int(bbox[1])
        x , _ = getCenterOfBBox(bbox)
        trianglePoints = np.array(
            [
                [x,y],
                [x-10,y-20],
                [x+10,y-20]
            ],np.int32
        )
        cv2.drawContours(frame,[trianglePoints],0,color,cv2.FILLED)
        #Desenhar borda
        cv2.drawContours(frame,[trianglePoints],0,(0,0,0),2)
        return frame

    def drawElipse(self,frame,bbox,color,trackID=None):
        y2 = int(bbox[3])        

        xCenter, yCenter = getCenterOfBBox(bbox)
        width = getBboxWidth(bbox)
        cv2.ellipse(frame,
                    (xCenter,y2), 
                    axes=(int(width),int(width * 0.35)), 
                    angle=0.0,
                    startAngle=-45,
                    endAngle=235, 
                    color=color, 
                    thickness=2,
                    lineType=cv2.LINE_4
                    )
        
        rectangleWidth = 40
        rectangleHeight = 20
        x1Rect = xCenter - rectangleWidth//2
        x2Rect = xCenter + rectangleWidth//2
        y1Rect = (y2 - rectangleHeight//2) + 15
        y2Rect = (y2 + rectangleHeight//2) + 15

        if trackID is not None:
            cv2.rectangle(frame,
                          (int(x1Rect),int(y1Rect)),
                          (int(x2Rect),int(y2Rect)),
                          color,
                          cv2.FILLED)
            x1Text = x1Rect + 12
            if trackID > 99:
                x1Text -= 10

            cv2.putText(frame,
                        str(trackID),
                        (int(x1Text),y1Rect + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,0,0),
                        2)
        return frame
    
    def drawTeamBallControl(self,frame,frameNum,teamBallControll):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = teamBallControll[:frameNum+1]
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Posse de bola time 1: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Posse de bola time 2: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame
    
    def drawAnnotations(self,videoFrames,tracks,teamBallControll):
        outputVideo = []
        for frameNum, frame in enumerate(videoFrames):
            frame = frame.copy()

            playerDict = tracks["players"][frameNum]
            ballDict = tracks["ball"][frameNum]
            refereeDict = tracks["referees"][frameNum]

            #Desenhar jogadores
            for trackID, player in playerDict.items():
                bbox = player["bbox"]
                color = player.get("teamColor",(0,0,255))
                frame = self.drawElipse(frame,bbox,color,trackID)
                if player.get("hasBall",False):
                    frame = self.drawTriangle(frame,bbox,(0,0,255))

            #Desenhar juiz
            for trackID, referee in refereeDict.items():
                bbox = referee["bbox"]
                frame = self.drawElipse(frame,bbox,(0,255,255))

            #Desenhar bola
            for trackID, ball in ballDict.items():
                bbox = ball["bbox"]
                frame = self.drawTriangle(frame,bbox,(0,255,0))
            

            # Posse da bola
            frame = self.drawTeamBallControl(frame,frameNum,teamBallControll)
            outputVideo.append(frame)
        print(f"Frames desenhados!")
        return outputVideo