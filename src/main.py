import glob
from args import getParsedArgs
from helpers import applyFrameMemory
from video_processing import blurAndWriteFrames, extractAndAddAudio, readVideoGetDetections

args = getParsedArgs()
import os

from ultralytics import YOLO
from torch import cuda

print(f"using {'cuda' if cuda.is_available() else 'cpu'} for detection")
import supervision as sv
from effects.blur import BlurAnnotator
import pickle
video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

inputFiles = []
isBatch = False
if os.path.isfile(args.input):
    inputFiles.append(args.input)
elif os.path.isdir(args.input):
    for file in glob.glob(os.path.join(args.input, "**"), recursive=True):
        inputFiles.append(file)
else:
    print(f"input {args.input} is not a file or directory")       

inputFiles = [file for file in inputFiles if os.path.splitext(file)[1].lower() in video_extensions]
isBatch = len(inputFiles) > 1
print(f"found {len(inputFiles)} video files")

for inputPath in inputFiles:
    print(f"reading video {inputPath}")

    if not os.path.isfile(inputPath):
        print(f"input file {inputPath} does not exist")
        continue

    if inputPath.endswith("_blurred.mp4"):
        print(f"input file {inputPath} is a blurred file, skipping")
        continue

    inputName = os.path.splitext(os.path.basename(inputPath))[0]
    outputPath = os.path.join(args.output, f"{inputName}_blurred.mp4")

    if not args.flat_output and isBatch:
        inputPathWithoutSrc = os.path.dirname( os.path.relpath(inputPath, args.input))
        
        outputPath = os.path.join(args.output, inputPathWithoutSrc, f"{inputName}_blurred.mp4")

    os.makedirs(os.path.dirname(outputPath), exist_ok=True)

    if os.path.isfile(outputPath):
        print(f"output file {outputPath} already exists, skipping")
        continue

    model = YOLO(model=args.model)
    detections = readVideoGetDetections(inputPath, model, args)

    #detections = pickle.load(open("detections.pkl", "rb"))
    if (args.frame_memory > 0):
        detections = applyFrameMemory(detections, args.frame_memory)

    blurAndWriteFrames(detections,
                       sv.BoxAnnotator() if args.no_blur else BlurAnnotator(),
                       inputPath, outputPath)
    
    extractAndAddAudio(inputPath, outputPath)
