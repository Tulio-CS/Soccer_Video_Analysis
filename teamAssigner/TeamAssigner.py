from sklearn.cluster import KMeans


class teamAssigner:
    def __init__ (self):
        self.teamColors = {}
        self.playerTeamDict = {}

    def assignTeamColor(self,frame,playerDetections):
        playerColors = []
        for _, playerDetection in playerDetections.items():
            bbox = playerDetection["bbox"]
            color = self.getColor(frame,bbox)
            playerColors.append(color)

        kmeans = KMeans(n_clusters=2,init="k-means++",n_init=1).fit(playerColors)

        self.kmeans = kmeans
        self.teamColors[1] = kmeans.cluster_centers_[0]
        self.teamColors[2] = kmeans.cluster_centers_[1]

    def getColor (self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        topHalfImage = image[0:int(image.shape[0]/2),:]

        #Clusterizar a imagem em 2
        image2d = topHalfImage.reshape((-1, 3))

        #Kmeans clustering
        kmeans = KMeans(n_clusters=2,init="k-means++",n_init=1).fit(image2d)

        labels = kmeans.labels_

        clusteredImage = labels.reshape(topHalfImage.shape[0], topHalfImage.shape[1])

        cornerClusters = [clusteredImage[0,0], clusteredImage[0,-1], clusteredImage[-1,0], clusteredImage[-1,-1]]
        nonPlayerCluster = max(set(cornerClusters), key=cornerClusters.count)
        playerCluster = 1 - nonPlayerCluster

        return kmeans.cluster_centers_[playerCluster]
    
    def playerTeam(self,frame,playerBbox,playerID):
        if playerID in self.playerTeamDict:
            return self.playerTeamDict[playerID]
        playerColor = self.getColor(frame,playerBbox)

        teamID = self.kmeans.predict(playerColor.reshape(1,-1))[0] + 1

        self.playerTeamDict[playerID] = teamID

        return teamID
