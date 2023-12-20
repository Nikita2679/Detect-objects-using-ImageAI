from imageai.Detection import ObjectDetection, VideoObjectDetection
import os

# detect objects on images and store them as new_objects.jpg

exec_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(exec_path, "retinanet_resnet50_fpn_coco-eeacb38b.pth"))
detector.loadModel()

list = detector.detectObjectsFromImage(
    input_image=os.path.join(exec_path, "objects.jpg"),
    output_image_path=os.path.join(exec_path, "new_objects.jpg"),
    minimum_percentage_probability=30,
    display_percentage_probability=False,
    display_object_name=False
)

# detect objects on video


execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolov3.pt"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(
    input_file_path=os.path.join(execution_path, "traffic.mp4"),
    output_file_path=os.path.join(execution_path, "traffic_detected"),
    frames_per_second=20, 
    log_progress=True
)
print(video_path)