import pyrealsense2 as rs
import json
import numpy as np

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)
for _ in range(5):
    frames = pipeline.wait_for_frames()

profile = pipeline.get_active_profile()
video_stream = profile.get_stream(rs.stream.color)
intr = video_stream.as_video_stream_profile().get_intrinsics()

# Build K matrix
K = [
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]
]

# Save to file
intrinsics = {
    "width": intr.width,
    "height": intr.height,
    "fx": intr.fx,
    "fy": intr.fy,
    "ppx": intr.ppx,
    "ppy": intr.ppy,
    "K": K
}
with open("intrinsics.json", "w") as f:
    json.dump(intrinsics, f, indent=2)

pipeline.stop()
print("✅ Saved RealSense intrinsics to 'intrinsics.json'")
