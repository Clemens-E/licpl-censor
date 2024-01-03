import argparse
from math import floor
from os import path


parser = argparse.ArgumentParser(
    description="licpl - blurring license plates and faces in dashcam videos"
)

parser.add_argument("--input", type=str, help="input video files", required=True)

parser.add_argument(
    "--frame-memory",
    type=int,
    help="number of detections (past and future) to consider for a frame",
    default=4,
)

parser.add_argument(
    "--batch", type=int, help="number of frames to process in one batch, keep it at default unless you tested performance", default=1
)

parser.add_argument(
    "--no-blur", help="do not blur, instead add annotations", action="store_true"
)

parser.add_argument(
    "--model", help="model to use for detections", default="best.pt"
)

parser.add_argument(
    "--multi", help="use experimental multiprocessing, most likely will not improve performance", default=False
)

parser.add_argument( 
    "--threads", type=int, help="select number of threads, use with --multi", default=floor(16 / 2)
)

parser.add_argument(
    "--save-training", help="save frames with low confidence for training", default=False
)

parser.add_argument(
    "--output", help="output video dir", required=True, 
)

parser.add_argument(
    "--flat-output", help="dont replicate the input folder structure", default=False
)

def getParsedArgs():
    argsObj = parser.parse_args()
    argsObj.output = path.normpath(argsObj.output)
    argsObj.input = path.normpath(argsObj.input)
    return argsObj

