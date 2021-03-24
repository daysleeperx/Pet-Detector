# Pet Detector
Pet Detector application uses [Tensorflow Lite](https://www.tensorflow.org/lite), [OpenCV](https://opencv.org/) and [Paho MQTT Client](https://pypi.org/project/paho-mqtt/) to detect objects and publish MQTT messages in order
to trigger automatic processes.

## Configuration
Settings can be configured in `config.ini` file.
Example configuration can be found in `config.ini.sample`.

## Installation
*Compatible only with Rasbpian Buster or Rasbpian Stretch*

start with `venv`
```bash


chmod +x venv_start.sh
./venv_start.sh
```

