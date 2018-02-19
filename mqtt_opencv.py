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
import pickle
import os

def bench(img, time_sent):
    img = base64.b64decode(img)
    frame = cv2.imdecode(np.fromstring(img, dtype=np.uint8), 1)
    frame = imutils.resize(frame, width=400)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)
 
    net.setInput(blob)
    detections = net.forward()

    # data = pickle.dumps(detections)

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
    # name = "image" + datetime.datetime.now().strftime('%H%M%S') + ".jpg"
    # cv2.imwrite(name, frame)
    # cv2.imwrite('image.jpg', frame)
    # print('done saving.')
    # Sending via MQTT
    img_str = cv2.imencode(".jpg", frame)
    send_data = base64.b64encode(img_str[1].tostring())
    # print(type(img_str[1].tostring()))
    
    base64_string = send_data.decode('utf-8')
    data = {"image": base64_string, "time_sent": time_sent}

    publish.single('hello/server', json.dumps(data), hostname="163.221.68.224")

def on_connect(mqttc, obj, flags, rc):
    print("rc: " + str(rc))


def on_message(mqttc, obj, msg):
    print('Time on first receive slave: {}'.format(datetime.datetime.now()))
    # time_received = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    # img = base64.b64decode(msg.payload)
    json_rec = msg.payload
    my_json = json_rec.decode('utf8').replace("'", '"')
    # print(my_json)
    # print(type(my_json))
    # print(json_rec)
    dmp = json.loads(my_json)
    bench(dmp['image'], dmp['time_sent'])

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
# make into argparse
mqttc.connect("163.221.68.224", 1883, 60)

conf = json.load(open('config.json'))
sub_to = 'hello/world' + conf['node']
mqttc.subscribe(sub_to, 0)

print("[INFO] loading model...")

dir_path = os.path.dirname(os.path.realpath(__file__))

prototxt = dir_path + '/MobileNetSSD_deploy.prototxt'
caffemodel = dir_path + '/MobileNetSSD_deploy.caffemodel'

net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

# net = None
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
# COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
COLORS = [[ 84.17340808, 195.2619167 ,  15.89690156],
       [250.12082477,  64.86789038, 127.30777312],
       [ 42.24136576,  58.37372156, 164.26309563],
       [223.06331711, 109.74376324, 155.80390287],
       [185.08709893, 226.65109253, 207.6055207 ],
       [236.07673731, 185.30827578, 202.81771854],
       [144.55106179, 144.33676777,  13.94532094],
       [187.12402101,  17.84717238, 169.79134966],
       [108.56370186,  11.93672853, 101.48437193],
       [242.67313174, 199.58060928, 105.16230962],
       [ 55.8721673 , 152.57844089,  10.81330649],
       [ 61.49715633, 202.01490572, 215.6031341 ],
       [ 46.84328774,  97.63950579,  45.02124015],
       [155.97362898, 170.12816067,  22.99799861],
       [ 82.28596664, 101.00173901, 133.31767819],
       [ 239.18937886, 246.96730454,  131.04242211],
       [157.12551971, 136.96627224, 219.19731213],
       [168.25525718,  46.68111693,  89.16807578],
       [ 41.48822204,  29.68208425, 244.29332197],
       [197.541347  , 106.32026389, 183.67652336],
       [203.44645816, 117.39418267, 127.80463932]]
mqttc.loop_forever()