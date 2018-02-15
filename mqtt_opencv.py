import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
import numpy as np
import cv2
import datetime
import base64

from imutils.video import FPS
import imutils
import time
import json

def bench(img):
    frame = cv2.imdecode(np.fromstring(img, dtype=np.uint8), 1)
    frame = imutils.resize(frame, width=400)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)
 
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # print('saving...')
    name = "image" + datetime.datetime.now().strftime('%H%M%S') + ".jpg"
    cv2.imwrite(name, frame)
    # cv2.imwrite('image.jpg', frame)
    # print('done saving.')
    # Sending via MQTT
    img_str = cv2.imencode(".jpg", frame)
    send_data = base64.b64encode(img_str[1].tostring())
    # print(type(img_str[1].tostring()))
    publish.single('hello/server', send_data, hostname="163.221.68.224")

def on_connect(mqttc, obj, flags, rc):
    print("rc: " + str(rc))


def on_message(mqttc, obj, msg):
    print('Time on first receive slave: {}'.format(datetime.datetime.now()))
    img = base64.b64decode(msg.payload)
    bench(img)

def on_publish(mqttc, obj, mid):
    print("mid: " + str(mid))


def on_subscribe(mqttc, obj, mid, granted_qos):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))


def on_log(mqttc, obj, level, string):
    print(string)

def on_disconnect(mqttc, obj, rc=0):
    print("Disconnected: " + str(rc))
    mqttc.loop_stop()


# If you want to use a specific client id, use
# mqttc = mqtt.Client("client-id")
# but note that the client id must be unique on the broker. Leaving the client
# id parameter empty will generate a random id for you.
mqttc = mqtt.Client()
mqttc.on_message = on_message
mqttc.on_connect = on_connect
mqttc.on_publish = on_publish
mqttc.on_subscribe = on_subscribe
mqttc.on_disconnect = on_disconnect

# Uncomment to enable debug messages
# mqttc.on_log = on_log
mqttc.connect("163.221.68.224", 1883, 60)
mqttc.subscribe("hello/world", 0)

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('/home/pi/virtualenvs/opencv/MobileNetSSD_deploy.prototxt', '/home/pi/virtualenvs/opencv/MobileNetSSD_deploy.caffemodel')
# net = None
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

mqttc.loop_forever()