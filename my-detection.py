from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput
import jetson.inference
import jetson.utils

net = detectNet("ssd-mobilenet-v2", threshold=0.5)
img, width, height = jetson.utils.loadImageRGBA("/home/nvidia/jetson-inference/examples/street.jpg")
detections = net.Detect(img, width, height)
for detection in detections:
    print(detection)
print("Got {:d} objects dected!".format(len(detections)))
save_path = "/home/nvidia/jetson-inference/examples/street_detect.jpg"
jetson.utils.saveImageRGBA(save_path, img, width, height)
