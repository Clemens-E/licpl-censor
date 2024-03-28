import multiprocessing

import imageio
import supervision as sv


def getFramesBufferMaxLength(args):
    return 30 if args.multi else args.batch


def getMaxCPUThreads():
    return multiprocessing.cpu_count()


import base64


def base64_encode_string(input_string):
    return base64.b64encode(input_string.encode()).decode()


def getFrameCount(inputFile):
    return imageio.get_reader(inputFile, "ffmpeg").count_frames()


def applyFrameMemory(detections: list[sv.Detections], memorySize: int):
    memoryFrames = []
    for index in range(len(detections)):
        mergedDetections = detections[index]

        for value2 in detections[index - memorySize:index + memorySize]:
            mergedDetections = sv.Detections.merge([mergedDetections, value2])

        if (index - memorySize) < 0:
            mergedDetections = sv.Detections.merge(
                [mergedDetections, detections[index]])

        memoryFrames.append(mergedDetections)
    return memoryFrames


lastFrameIndex = 0


def getLowestConf(detc):
    proc = [d for d in detc]
    lowestConf = 1
    for detection in proc:
        print(detection[2])
        if detection[2] < lowestConf:
            lowestConf = detection[2]

    return lowestConf


def checkLowConf(processedFrames, args):
    for index in range(len(processedFrames)):
        if getLowestConf(processedFrames[index]) < 0.30 and \
                (lastFrameIndex + 30) < index:
            # append index to file
            with open(f"low_confidence_{base64_encode_string(args.input)}.txt",
                      "a") as f:
                f.write(f"{index}\n")
            lastFrameIndex = index
