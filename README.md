# Pet Detector
Pet Detector application uses [Tensorflow Lite](https://www.tensorflow.org/lite), [OpenCV](https://opencv.org/) and [Paho MQTT Client](https://pypi.org/project/paho-mqtt/) to detect objects and publish MQTT messages in order
to trigger automatic processes.

## Configuration
Settings can be configured in `config.ini` file.
Example configuration can be found in `config.ini.sample`.

## Installation
*Compatible only with Rasbpian Buster or Rasbpian Stretch*

Camera needs to be enabled in `raspi-config`

Raspberry Pi needs to be updated
```bash
sudo apt-get update
sudo apt-get dist-upgrade
```

start with `venv`
```bash
chmod +x venv_start.sh
./venv_start.sh
```

## Edge TPU
Install necessary dependencies
```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std
```

Change following lines in `config.ini`
```ini
tflite_file = edgetpu.tflite
edgetpu = true
```

