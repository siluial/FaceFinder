from imutils.video import VideoStream
import time
import imutils


def setup():
    global vs
    vs = VideoStream(src=0).start()
    time.sleep(2.0)


def grab():
    frame = vs.read()
    return imutils.resize(frame, width=400)