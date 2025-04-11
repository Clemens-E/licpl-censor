import os
import cv2
import imageio
from tqdm import tqdm
from detection import getDetectionsBasedOnArgs
from subprocess import DEVNULL, check_call
import supervision as sv
from helpers import getFrameCount, getFramesBufferMaxLength


def blurAndWriteFrames(detections,
                       annotator,
                       inputPath,
                       outputPath,
                       args,
                       tqdm=tqdm):
    if (os.path.isfile(outputPath)):
        print(f"output video {outputPath} already exists, removing")
        os.remove(outputPath)

    tmpReader = imageio.get_reader(inputPath)
    metaData = tmpReader.get_meta_data()
    originalFps = metaData["fps"]
    frameCount = getFrameCount(inputPath)

    tmpReader.close()

    with imageio.get_writer(outputPath,
                            codec="libx264",
                            macro_block_size=None,
                            quality=args.quality,
                            fps=originalFps) as writer:

        videoReader = imageio.imiter(inputPath)
        count = 0
        with tqdm(total=frameCount) as pbar:
            for frame in videoReader:
                annotated_frame = annotator.annotate(
                    scene=frame,
                    detections=detections[count],
                )
                writer.append_data(annotated_frame.astype("uint8"))

                count += 1
                pbar.update(1)


def extractAndAddAudio(inputPath, outputPath, log=print):
    outputPathBase = os.path.dirname(outputPath)
    tmpAudio = os.path.join(outputPathBase, f"{os.urandom(8).hex()}.aac")
    tmpVideo = os.path.join(outputPathBase, f"{os.urandom(8).hex()}.mp4")
    try:
        check_call(
            ["ffmpeg", "-i", inputPath, "-vn", tmpAudio],
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
        log(f"adding audio to output video")
        check_call(
            [
                "ffmpeg", "-i", tmpAudio, "-i", outputPath, "-c:v", "copy",
                "-c:a", "aac", "-strict", "experimental", tmpVideo
            ],
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
        os.remove(tmpAudio)
        os.remove(outputPath)
        os.rename(tmpVideo, outputPath)
    except Exception as e:
        log(f"An error occurred while adding audio to the output video")
        log(f"leaving video without audio")
        if os.path.isfile(tmpAudio):
            os.remove(tmpAudio)


def readVideoGetDetections(inputPath, yol, args, tqdm=tqdm) -> list[sv.Detections]:
    frameCount = getFrameCount(inputPath)
    videoReader = imageio.imiter(inputPath)
    count = 0
    framesBuffer = []
    processedFrames = []

    with tqdm(total=frameCount) as pbar:
        for frame in videoReader:
            framesBuffer.append(
                cv2.cvtColor(frame.astype("uint8"), cv2.COLOR_BGR2RGB))

            if len(framesBuffer) > getFramesBufferMaxLength(args):
                processedFrames.extend(
                    getDetectionsBasedOnArgs(yol=yol,
                                             frames=framesBuffer,
                                             args=args))
                pbar.update(len(framesBuffer))
                framesBuffer = []
            count += 1

        if len(framesBuffer) > 0:
            processedFrames.extend(
                getDetectionsBasedOnArgs(yol=yol,
                                         frames=framesBuffer,
                                         args=args))
            pbar.update(len(framesBuffer))

    return processedFrames
