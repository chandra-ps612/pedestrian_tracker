import numpy as np
import imutils
import time
import cv2

onnx_model_path= 'C:\\Users\\lav singh\\OneDrive\\Desktop\\label_crossing_project\\yolov5x.onnx'
dataset= 'C:\\Users\\lav singh\\OneDrive\\Desktop\\label_crossing_project\\coco.names'
videoPath= 'C:\\Users\\lav singh\\OneDrive\\Desktop\\label_crossing_project\\label_crossing.MOV'
output_videoPath= 'C:\\Users\\lav singh\\OneDrive\\Desktop\\label_crossing_project\\out1.avi'

# Constants:
CONF_THRESHOLD=0.45
NMS_THRESHOLD=0.4
SCORE_THRESHOLD=0.5

from sort import *
pedestrain_tracker = Sort()
memory = {}


# The list classes
classes=[]
with open(dataset, 'r') as f:
    classes=[line.strip() for line in f.readlines()]

# Initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

# Loading yolov5m.onnx model
net = cv2.dnn.readNetFromONNX(onnx_model_path)
ln=net.getLayerNames() # To get all the name of all layers of the network
for i in net.getUnconnectedOutLayers():
    output_layers= ln[i-1]

# Initialize the video stream, pointer to output video file, and frame dimensions
cap = cv2.VideoCapture(videoPath)
create = None
frameIndex = 0

# Try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(cap.get(prop))
    print(f'Total frames in video is {total}')

# An error occurred while trying to determine the total number of frames in the video file
except:
    print('Could not determine # of frames in video')
    print('No approx. completion time can be provided')
    total = -1

# Loop over frames from the video file stream
while True:
    _, frame=cap.read()
    frame=cv2.resize(frame, (640, 640), fx=None, fy=None)
    # Pre-Processing
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    outputs=net.forward(output_layers)
    end = time.time()
    # Get box dimensions
    boxes=[]
    confs=[]
    class_ids=[]

    for output in outputs:
        for detection in output:
            conf=detection[4]
            if conf > CONF_THRESHOLD:
                classes_scores=detection[5:]
                class_id=np.argmax(classes_scores)
                if (classes_scores[class_id]>SCORE_THRESHOLD):
                    if (class_id==0):
                        # Object detected
                        center_x=int(detection[0])
                        center_y=int(detection[1])
                        w=int(detection[2])
                        h=int(detection[3])
                        # Rectangle co-ordinates
                        x=int(center_x-w/2)
                        y=int(center_y-h/2)
                        box=np.array([x, y, w, h])
                        boxes.append(box)
                        confs.append(conf)
                        class_ids.append(class_id)
    indices=cv2.dnn.NMSBoxes(boxes, confs, SCORE_THRESHOLD, NMS_THRESHOLD)
    dets = []
    if len(indices) > 0:
        # Loop over the indices we are keeping
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x+w, y+h, confs[i]])

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)
    tracks = pedestrain_tracker.update(dets)

    boxes = []
    index_IDs = []
    c = []
    previous = memory.copy()
    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        index_IDs.append(int(track[4]))
        memory[index_IDs[-1]] = boxes[-1]

    if len(boxes) > 0:
        i = int(0)
        for box in boxes:
            # Extract the bounding box coordinates
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))

            color = [int(c) for c in COLORS[index_IDs[i] % len(COLORS)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 1)

            if index_IDs[i] in previous:
                previous_box = previous[index_IDs[i]]
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
            
            cv2.putText(frame, f'ID:{index_IDs[i]}', (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.4, color, 1)
            i += 1

    cv2.imshow('result', frame)
    # Saving post-processed video
    if create is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        create = cv2.VideoWriter(output_videoPath, fourcc, 5, (frame.shape[1], frame.shape[0]), True)

        # Some information about processing single frame
        if total > 0:
           elap = (end - start)
           print(f'Single frame took {elap} seconds')
           print(f'Estimated total time to finish:{elap*total} seconds')

    # Write the output frame to disk
    create.write(frame)
    key=cv2.waitKey(1) # It'll generate a new frame after every 1 ms.
    if key==ord('q'):
        break
    # Increase frame index
    frameIndex += 1

    if frameIndex >= 340:
        print('Cleaning up...')
        create.release()
        cap.release()
        exit()

# Release the file pointers
print('Cleaning up...')
create.release()
cap.release()