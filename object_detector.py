from dotenv import load_dotenv
from google.cloud import videointelligence

load_dotenv()
# Object tracking algorithm using Google Video Intelligence API.

video_client = videointelligence.VideoIntelligenceServiceClient()
features = [videointelligence.Feature.OBJECT_TRACKING]

file_path = './Test_atm.mp4'

with open(file_path,'rb') as file :
    input_content = file.read() 

operation = video_client.annotate_video(
    request={"features": features, "input_content" : input_content }
)

print("\nProcessing video for object annotations")

result = operation.result(timeout=5000)
print("\nFinished processing.\n")

object_annotations = result.annotation_results[0].object_annotations


object_annotation = object_annotations[0]
print("Entity description: {}".format(object_annotation.entity.description))

if object_annotation.entity.entity_id:
    print("Entity id {}:".format(object_annotation.entity.entity_id) )

print("Confidence: {}".format(object_annotation.confidence))

# Here we print only the bounding box of the first frame in this segment
frame = object_annotation.frames[0]
box = frame.normalized_bounding_box
print(
    "Time offset of the first frame: {}s".format(
        frame.time_offset.seconds + frame.time_offset.microseconds / 1e6
    )
)
print("Bounding box position:")
print("\tleft  : {}".format(box.left))
print("\ttop   : {}".format(box.top))
print("\tright : {}".format(box.right))
print("\tbottom: {}".format(box.bottom))
print("\n")
