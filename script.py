import numpy as np
import tensorflow as tf
import cv2
import time
from sklearn.metrics import pairwise
from imutils.video import FPS

from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

font = cv2.FONT_HERSHEY_TRIPLEX


utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile
PATH_TO_LABELS = './data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# model can be downloaded by http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
model_name = 'ssdlite_mobilenet_v2_coco_2018_05_09' 

model_dir =  "./models/" + model_name + "/saved_model"
detection_model = tf.saved_model.load(str(model_dir))
detection_model = detection_model.signatures['serving_default']

crash_count_frames = 0
def estimate_collide(output_dict,height,width,image_np):
  global crash_count_frames
  vehicle_crash = 0
  max_curr_obj_area = 0
  centerX = centerY = 0
  details = [0 , 0 , 0 , 0]
  for ind,scr in enumerate(output_dict['detection_classes']):
    if scr==2 or scr==3 or scr==4 or scr==6 or scr==8:
      ymin, xmin, ymax, xmax = output_dict['detection_boxes'][ind]
      score = output_dict['detection_scores'][ind]
      if score>0.5:
        obj_area = int((xmax - xmin)*width * (ymax - ymin)*height)
        if obj_area > max_curr_obj_area:
          max_curr_obj_area = obj_area
          details = [ymin, xmin, ymax, xmax]

  centerX , centerY = (details[1] + details[3])/2 , (details[0] + details[2])/2
  if max_curr_obj_area>70000:
    if (centerX < 0.2 and details[2] > 0.9) or (0.2 <= centerX <= 0.8) or (centerX > 0.8 and details[2] > 0.9):
      vehicle_crash = 1
      crash_count_frames = 15

  if vehicle_crash == 0:
    crash_count_frames = crash_count_frames - 1
    
  if crash_count_frames > 0:
    if max_curr_obj_area <= 100000:
      cv2.putText(image_np,"WARNING !!!" ,(100,100), font, 4, (255,0,0),3,cv2.LINE_AA)


def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  input_tensor = tf.convert_to_tensor(image)
  input_tensor = input_tensor[tf.newaxis,...]

  output_dict = model(input_tensor)

  num_detections = int(output_dict.pop('num_detections'))


  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
  return output_dict


def show_inference(model, image_path):
  image_np = np.array(image_path)
  height,width,channel = image_np.shape

  output_dict = run_inference_for_single_image(model, image_np)
  estimate_collide(output_dict,height,width,image_np)

  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

  return image_np

filename = input("Enter the video name: ");
print ("Wait until the video is processed...");

cap=cv2.VideoCapture('./videos/'+filename+'.mp4')
time.sleep(2.0)

cap.set(1,0)

fps = FPS().start()

ctt = 0
while True:
    (grabbed, frame) = cap.read()
    frame = frame[ :-150, : , :]
    ctt = ctt + 1
    if ctt==3334:
      break
    frame=show_inference(detection_model, frame)


    cv2.imshow("version", frame)
    fps.update()

    key=cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break
        
fps.stop()
cap.release()
cv2.destroyAllWindows()