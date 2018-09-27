import cv2


def setup():
    global net
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")


def detect(frame):
    # convert frame to blob sized to match the network model
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # push blob through network and get detections
    net.setInput(blob)
    return net.forward()
