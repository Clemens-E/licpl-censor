import argparse

parser = argparse.ArgumentParser(
    description="licpl - blurring license plates and faces in dashcam videos"
)

parser.add_argument("--input", type=str, help="input video file", required=True)
parser.add_argument(
    "--frame-memory",
    type=int,
    help="number of detection to remember from the previous and next frames",
    default=4,
)
parser.add_argument(
    "--kernel-size", type=int, help="kernel size for blurring", default=10
)
parser.add_argument(
    "--batch", type=int, help="number of frames to process in one batch", default=2
)
parser.add_argument(
    "--no-blur", help="do not blur, instead add detections", action="store_true"
)


def getParsedArgs():
    return parser.parse_args()

