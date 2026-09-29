import traceback

print("Running vision engine on test video...")
try:
    from vision_engine import process_video_vision
    res = process_video_vision('test_face.mp4', 'dummy_out.mp4')
    print("Vision engine completed:", res)
except Exception as e:
    print("Exception occurred!")
    traceback.print_exc()
