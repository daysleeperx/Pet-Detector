import configparser
import os
import time
from datetime import datetime
from typing import List

import cv2 as cv
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

from mqtt_client import connect_mqtt, publish
from video_stream import VideoStream

config = configparser.ConfigParser(os.environ)
config.read("config.ini")

MODEL_NAME = config.get('tensorflow', 'model_dir')
TFLITE_FILE = config.get('tensorflow', 'tflite_file')
LABELMAP = config.get('tensorflow', 'labels')
MIN_THRESHOLD = float(config.get('tensorflow', 'threshold'))
RES_WIDTH, RES_HEIGHT = config.get('webcam', 'resolution').split('x')
IMG_WIDTH, IMG_HEIGHT = int(RES_WIDTH), int(RES_HEIGHT)
EDGE_TPU = config.getboolean('tensorflow', 'edgetpu')

CWD_PATH = os.getcwd()
PATH_TO_MODEL = os.path.join(CWD_PATH, MODEL_NAME, TFLITE_FILE)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP)


def load_labels(path: str) -> List[str]:
    """Load labels from file."""
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()][1:]


def init_interpreter(path_to_model: str, edge_tpu=False) -> Interpreter:
    """Initialize the interpreter."""
    log(f'Path to model: {path_to_model}')
    delegates = [load_delegate('libedgetpu.so.1.0')] if edge_tpu else None
    interpreter = Interpreter(model_path=path_to_model, experimental_delegates=delegates)
    return interpreter


def get_input(frame, height, width):
    """Resize and return input data."""
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_resized = cv.resize(frame_rgb, (width, height))
    return np.expand_dims(frame_resized, axis=0)


def detect_objects(input_data, interpreter) -> tuple:
    """Retrieve detection results."""
    tensor_index = interpreter.get_input_details()[0]['index']
    interpreter.set_tensor(tensor_index, input_data)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    return boxes, classes, scores


def draw_label(frame, label, xmin, ymin):
    """Draw object label."""
    label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    label_ymin = max(ymin, label_size[1] + 10)
    cv.rectangle(frame, (xmin, label_ymin - label_size[1] - 10), (xmin + label_size[0], label_ymin + base_line - 10),
                 (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (xmin, label_ymin - 7), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def draw_bounding_box(frame, xmax, xmin, ymax, ymin):
    """Draw bounding box."""
    cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)


def annotate_objects(boxes, classes, scores, frame, labels, threshold):
    """
    Loop over all detections and draw detection box if confidence is above minimum threshold.

    Draw bounding box and label. Interpreter can return coordinates that are outside of image dimensions,
    need to force them to be within image using max() and min()
    """
    annotated_objects = []
    for i in range(len(scores)):
        if threshold < scores[i] <= 1.0:
            ymin, xmin = int(max(1, (boxes[i][0] * IMG_HEIGHT))), int(max(1, (boxes[i][1] * IMG_WIDTH)))
            ymax, xmax = int(min(IMG_HEIGHT, (boxes[i][2] * IMG_HEIGHT))), int(min(IMG_WIDTH, (boxes[i][3] * IMG_WIDTH)))

            object_name = labels[int(classes[i])]
            label = f'{object_name}: {int(scores[i] * 100)}%'

            draw_bounding_box(frame, xmax, xmin, ymax, ymin)
            draw_label(frame, label, xmin, ymin)
            annotated_objects.append(object_name)

    return annotated_objects


def log(message: str):
    """Logging utility method."""
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f'[{current_time}] {message}')


def run_detection():
    """Main detection method."""
    labels = load_labels(PATH_TO_LABELS)
    interpreter = init_interpreter(PATH_TO_MODEL, EDGE_TPU)
    interpreter.allocate_tensors()
    (_, height, width, _) = interpreter.get_input_details()[0]['shape']

    videostream = VideoStream(resolution=(IMG_WIDTH, IMG_HEIGHT), framerate=30)
    videostream.start()
    time.sleep(1)

    frame_rate_calc = 1
    freq = cv.getTickFrequency()

    mqtt_client = connect_mqtt(log)
    mqtt_client.loop_start()

    cat_counter = 0

    while True:
        t1 = cv.getTickCount()
        frame = videostream.read().copy()

        input_data = get_input(frame, height, width)
        (boxes, classes, scores) = detect_objects(input_data, interpreter)

        annotated = annotate_objects(boxes, classes, scores, frame, labels, MIN_THRESHOLD)

        if 'cat' in annotated:
            cat_counter += 1

        if cat_counter >= 20:
            log(f'CAT DETECTED, cat count: {cat_counter}')
            publish(mqtt_client, 'volumio/playback/playPlaylist', 'Cat Music', log)
            cat_counter = 0

        cv.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0),2, cv.LINE_AA)
        cv.imshow('Pet detector', frame)

        t2 = cv.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

        if cv.waitKey(1) == ord('q'):
            break

    cv.destroyAllWindows()
    videostream.stop()


if __name__ == '__main__':
    run_detection()
