
import os
from dotenv import load_dotenv
from google.cloud import videointelligence

from helpers.detector import detect_suspicious_simple


## Algorithm for detection using Google Video Intelligence API.
def load_video_content(video_path='./Test_atm.mp4'):
    print("load_video_content() called\n")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    with open(video_path, 'rb') as media_file:
        print(f"Loading video: {video_path}")
        return media_file.read()

def configure_people_detection(
    include_bounding_boxes= True, # Bounding box to identify people
    include_attributes=True,
    include_pose_landmarks=True 
    ):
    print("configure_people_detection() called\n")
    config = videointelligence.PersonDetectionConfig(
        include_bounding_boxes= include_bounding_boxes,
        include_attributes= include_attributes,
        include_pose_landmarks=include_pose_landmarks
    )
    return videointelligence.VideoContext(person_detection_config=config)

def process_video(video_client, input_content, context):
    operation = video_client.annotate_video(
        request={
            "features": [videointelligence.Feature.PERSON_DETECTION],
            "input_content": input_content,
            "video_context": context,
        }
    )
    print("\nVideo processing for annotations to detect suspicious actions of people.")
    result = operation.result(timeout=300)
    print("Finished processing\n")
    return result.annotation_results[0]


def print_people_detection_annotations(annotation_result):
    for person in annotation_result.person_detection_annotations:
        print("Person detected")
        for track in person.tracks:
            start_s = track.segment.start_time_offset.seconds + track.segment.start_time_offset.microseconds/1e6
            end_s   = track.segment.end_time_offset.seconds   + track.segment.end_time_offset.microseconds/1e6
            print(f"  segment: {start_s:.2f}s to {end_s:.2f}s")


            # "Suspicious activities" detector
            intervals = detect_suspicious_simple(track)
            if intervals:
                for a, b in intervals:
                    print(f"[Suspicious Hand Motion Detected] {a:.2f}s -> {b:.2f}s")
            else:
                print("No suspicious hand motion detected.")


def main():
    try:
        load_dotenv()
        print("✓ Environment variables loaded")
        print("Starting analysis...\n")
        input_content = load_video_content()
        print("✓ Video loaded")
        video_client = videointelligence.VideoIntelligenceServiceClient()
        print("✓ Video Intelligence client activated")
        context = configure_people_detection()
        annotation_result = process_video(video_client, input_content, context)
        print_people_detection_annotations(annotation_result)
       
    except FileNotFoundError as e:
        print(f"File Error: {e}")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
