"""VideoStream."""

import cv2 as cv
from threading import Thread


class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(640, 480), framerate=30):
        """Initialize the PiCamera and the camera image stream."""
        self.stream = cv.VideoCapture(0)
        self.stream.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(3, resolution[0])
        self.stream.set(4, resolution[1])

        (_, self.frame) = self.stream.read()

        self.stopped = False

    def start(self):
        """Start the thread that reads frames from the video stream."""
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        """Keep looping indefinitely until the thread is stopped."""
        while True:
            if self.stopped:
                self.stream.release()
                return

            (_, self.frame) = self.stream.read()

    def read(self):
        """Return the most recent frame."""
        return self.frame

    def stop(self):
        """Indicate that the camera and thread should be stopped."""
        self.stopped = True
