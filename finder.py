import grabber
import detector
import cv2
import numpy as np


confidence_threshold = 0.5


def draw_detection_boxes(frame, startX, startY, endX, endY, confidence):
    # draw the bounding box of the face along with the associated
    # probability
    text = "{:.2f}%".format(confidence * 100)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    return frame


def main():
    grabber.setup()
    detector.setup()

    seen_faces = []

    while True:
        frame = grabber.grab()
        detections = detector.detect(frame)

        (h, w) = frame.shape[:2]

        # loop over the detections
        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]
            if confidence < confidence_threshold:
                continue

            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]

            is_seen_before(seen_faces, face)

            frame = draw_detection_boxes(frame, startX, startY, endX, endY, confidence)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()
