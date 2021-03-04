"""MQTT client utilities."""

import configparser
import os
import secrets

from paho.mqtt import client as mqtt

config = configparser.ConfigParser(os.environ)
config.read("config.ini")

CLIENT_ID = f'python-mqtt-{secrets.token_hex(16)}'
MQTT_BROKER = config.get('mqtt', 'broker')


def connect_mqtt(logger=lambda msg: print(msg)) -> mqtt.Client:
    """Create client and connect to broker."""

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            logger('Connected to MQTT Broker!')
        else:
            logger(f'Failed to connect, return code {rc}\n')

    client = mqtt.Client(CLIENT_ID)
    client.on_connect = on_connect
    client.connect(MQTT_BROKER)
    return client


def publish(client, topic, message, logger=lambda msg: print(msg)):
    """Publish message to specified topic."""
    result = client.publish(topic, message)
    status = result[0]

    if status == 0:
        logger(f"Sent `{message}` to topic `{topic}`")
    else:
        logger(f"Failed to send {message} to topic {topic}")
