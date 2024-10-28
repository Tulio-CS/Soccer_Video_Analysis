from utils import readVideo, saveVideo
from tracker import Tracker
from teamAssigner import teamAssigner
from playerBallAssigner import PlayerBallAssigner
import cv2
import numpy as np
from cameraMovementEstimator import CameraMovementEstimator

def main():
    videoFrames = readVideo("inputVideos/08fd33_4.mp4")

    tracker = Tracker("models/yoloV5trained.pt")

    tracks = tracker.getObjectTracks(videoFrames,
                                     readFromStub=True,
                                     stubPath="stubs/trackStubs.pkl")

    # Adiciona posicao do objeto ao track
    tracker.addPositionToTrack(tracks)



    cameraMovementEstimator = CameraMovementEstimator(videoFrames[0])
    cameraMovementPerFrame = cameraMovementEstimator.getCameraMovement(videoFrames,
                                                                       readFromStubs=True,
                                                                       stubPath="stubs/cameraMovementStubs.pkl")

    cameraMovementEstimator.adjustPositionsToTracks(tracks,cameraMovementPerFrame)
    #Interpolar posicao da bola
    tracks["ball"] = tracker.interpolateBall(tracks["ball"])

    # Atribuir um time para jogador
    team_Assigner = teamAssigner()
    team_Assigner.assignTeamColor(videoFrames[0],tracks["players"][0])

    for frameNum, playerTrack in enumerate(tracks["players"]):
        for playerID, track in playerTrack.items():
            teamID = team_Assigner.playerTeam(videoFrames[frameNum],track["bbox"],playerID)
            tracks["players"][frameNum][playerID]["team"] = teamID
            tracks["players"][frameNum][playerID]["teamColor"] = team_Assigner.teamColors[teamID]

    # Assign ball to player
    playerAssigner = PlayerBallAssigner()
    teamBallControll = []
    for frameNum, playerTrack in enumerate(tracks["players"]):
        ballBbbox = tracks["ball"][frameNum][1]["bbox"]
        assignedPlayer = playerAssigner.assignBallToPlayer(playerTrack,ballBbbox)
        if assignedPlayer != -1:
            tracks["players"][frameNum][assignedPlayer]["hasBall"] = True
            teamBallControll.append(tracks["players"][frameNum][assignedPlayer]["team"])
        else:
            teamBallControll.append(teamBallControll[-1])

    teamBallControll = np.array(teamBallControll)
    outputVideoFrames = tracker.drawAnnotations(videoFrames,tracks,teamBallControll)
    outputVideoFrames = cameraMovementEstimator.drawCameraMovement(outputVideoFrames, cameraMovementPerFrame)
    saveVideo(outputVideoFrames, "outputVideos/output.avi")
    print("Done!")

if __name__ == "__main__":
    main()