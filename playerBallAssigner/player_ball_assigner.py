
import sys
sys.path.append("../")
from utils import getCenterOfBBox, measureDistance

class PlayerBallAssigner():
    def __init__(self):
        self.maxPlayerBallDistance = 70
    
    def assignBallToPlayer(self,players,ballBbox):
        ballPosition = getCenterOfBBox(ballBbox)

        minDistance = 999999
        assignedPlayer = -1
        for playerID, player in players.items():
            playerBbox = player["bbox"]
            distanceLeft = measureDistance((playerBbox[0],playerBbox[-1]),ballPosition)
            distanceRight = measureDistance((playerBbox[2],playerBbox[-1]),ballPosition)
            distance = min(distanceLeft,distanceRight)

            if distance < minDistance and distance < self.maxPlayerBallDistance:
                minDistance = distance
                assignedPlayer = playerID

        return assignedPlayer