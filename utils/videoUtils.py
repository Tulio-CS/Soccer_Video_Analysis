import cv2

def readVideo(videoPath):
    cap = cv2.VideoCapture(videoPath)
    frames = []
    while True: 
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    print(f"Video lido com sucesso!")
    return frames

def saveVideo(frames, outputVideoPath):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outputVideoPath, fourcc, 24, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    print(f"Video salvo com sucesso!")
    out.release()