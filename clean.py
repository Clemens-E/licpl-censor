from math import pi
from args import getParsedArgs

args = getParsedArgs()
import os

# check if input file exists
if not os.path.isfile(args.input):
    print(f"input file {args.input} does not exist")
    exit(1)

from ultralytics import YOLO
from torch import cuda
print(f"using {'cuda' if cuda.is_available() else 'cpu'} for detection")
import supervision as sv
import cv2
import imageio
from tqdm import tqdm
from helpers import getDetectionsFromFrames


# read images from video
frameCount = imageio.get_reader(args.input, "ffmpeg").count_frames()
videoReader = imageio.imiter(args.input)
yol = YOLO(model="./best-m.pt")
count = 0
framesBuffer = []
processedFrames = []

print(f"processing {frameCount} frames")
with tqdm(total=frameCount) as pbar:
    for frame in videoReader:
        framesBuffer.append(cv2.cvtColor(frame.astype("uint8"), cv2.COLOR_BGR2RGB))

        if len(framesBuffer) > args.batch:
            processedFrames.extend(getDetectionsFromFrames(yol, framesBuffer))
            pbar.update(len(framesBuffer))
            framesBuffer = []
        count += 1

    if len(framesBuffer) > 0:
        processedFrames.extend(getDetectionsFromFrames(yol, framesBuffer))
        pbar.update(len(framesBuffer))


# merge detections from previous and next frames
memoryFrames = []
if args.frame_memory > 0:
    for index in range(len(processedFrames)):
        mergedDetections = sv.Detections.empty()
        mergedDetections = sv.Detections.merge(
            [mergedDetections, processedFrames[index]]
        )

        for value2 in processedFrames[
            index - args.frame_memory : index + args.frame_memory
        ]:
            mergedDetections = sv.Detections.merge([mergedDetections, value2])

        memoryFrames.append(mergedDetections)
else:
    memoryFrames = processedFrames

blur_annotator = sv.BoxAnnotator() if args.no_blur else sv.BlurAnnotator()
inputName = os.path.basename(args.input).split(".")[0]
randomString = os.urandom(8).hex()
tmpVideo = f"{randomString}.mp4"
tmpAudio = f"{randomString}.aac"
outputVideo = os.path.join(os.path.dirname(args.input),f"{inputName}_blurred.mp4")
print(f" {inputName}  {outputVideo}")
print(f"blurring frames")
# create new opencv video to add frames
with imageio.get_writer(
    tmpVideo,
    codec="libx264",
    macro_block_size=None,
    fps=imageio.get_reader(args.input).get_meta_data()["fps"],
) as writer:

    videoReader = imageio.imiter(args.input)
    count = 0
    with tqdm(total=frameCount) as pbar:
        for frame in videoReader:
            annotated_frame = blur_annotator.annotate(
                scene=frame,
                detections=memoryFrames[count],
            )
            frame_blurred_rgb = cv2.cvtColor(
                annotated_frame.astype("uint8"), cv2.COLOR_RGB2BGR
            )
            writer.append_data(annotated_frame.astype("uint8"))
            count += 1
            pbar.update(1)

print(f"extracting audio from source video")
from subprocess import DEVNULL, STDOUT, check_call
if (os.path.isfile(outputVideo)):
    print(f"output video {outputVideo} already exists, removing")
    os.remove(outputVideo)
check_call(["ffmpeg", "-i", args.input, "-vn", "-acodec", "copy", tmpAudio], stdout=DEVNULL, stderr=STDOUT)
print(f"adding audio to output video")
check_call(["ffmpeg", "-i", tmpAudio, "-i", tmpVideo, "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", outputVideo], stdout=DEVNULL, stderr=STDOUT)
os.remove(tmpVideo)
os.remove(tmpAudio)
