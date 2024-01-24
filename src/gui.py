from functools import partial
from glob import glob
from os import makedirs, path
from torch import cuda
from customtkinter import filedialog
import customtkinter as ct
from tqdm.tk import tqdm
import threading

ct.set_appearance_mode("System")  # Modes: system (default), light, dark
ct.set_default_color_theme("dark-blue")

app = ct.CTk()
app.title("LICPL—Censor")
app.geometry("800x400")

# Create a frame for the inputs
input_frame = ct.CTkFrame(app)
input_frame.pack(side="left", fill="both", expand=False, padx=10)

# Create a frame for the text field
text_frame = ct.CTkFrame(app)
text_frame.pack(side="right", fill="both", expand=True)


progress_bar = ct.CTkProgressBar(text_frame, )
progress_bar.set(0)
progress_bar.pack(fill="x")

cuda_status_label = ct.CTkLabel(
    input_frame,
    text=f"using {'gpu' if cuda.is_available() else 'cpu'}",
    fg_color="green" if cuda.is_available() else "red",
)
cuda_status_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')

input_label = ct.CTkLabel(input_frame, text="Input Folder")
input_label.grid(row=1, column=0, padx=10, pady=10)
input_entry = ct.CTkEntry(input_frame)
input_entry.grid(row=1, column=1, padx=10, pady=10, sticky='e')
input_button = ct.CTkButton(
    input_frame,
    text="🔍",
    width=5,
    command=lambda: input_entry.insert(0, filedialog.askdirectory()))
input_button.grid(row=1, column=2, padx=5, pady=5, sticky='w')

output_label = ct.CTkLabel(input_frame, text="Output Folder")
output_label.grid(row=2, column=0, padx=10, pady=10)
output_entry = ct.CTkEntry(input_frame)
output_entry.grid(row=2, column=1, padx=10, pady=10)
output_button = ct.CTkButton(
    input_frame,
    text="🔍",
    width=5,
    command=lambda: output_entry.insert(0, filedialog.askdirectory()))
output_button.grid(row=2, column=2, padx=5, pady=5, sticky='w')

model_label = ct.CTkLabel(input_frame, text="Model File")
model_label.grid(row=3, column=0, padx=10, pady=10)
model_entry = ct.CTkEntry(input_frame)
model_entry.grid(row=3, column=1, padx=10, pady=10)
model_button = ct.CTkButton(
    input_frame,
    text="🔍",
    width=5,
    command=lambda: model_entry.insert(
        0, filedialog.askopenfilename(filetypes=[("Model File", "*.pt")])))
model_button.grid(row=3, column=2, padx=5, pady=5, sticky='w')

# Add a text field to the text frame
text_field = ct.CTkTextbox(text_frame)
text_field.pack(fill="both", expand=True)
text_field.tag_config("green", foreground="green")

video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']


def checkEmptyFields(entry):
    if entry.get() == "":
        entry.focus()
        return True
    return False


def runThread():
    r = checkEmptyFields(input_entry)
    r = r or checkEmptyFields(output_entry)
    r = r or checkEmptyFields(model_entry)
    if r:
        return
    threading.Thread(target=runDetection).start()


def runDetection():
    inputFiles = []
    for file in glob(path.join(input_entry.get(), "**"), recursive=True):
        inputFiles.append(file)
    inputFiles = [
        file for file in inputFiles
        if path.splitext(file)[1].lower() in video_extensions
    ]
    text_field.insert("end", f"found {len(inputFiles)} video files\n")
    if len(inputFiles) == 0:
        return

    from video_processing import blurAndWriteFrames, extractAndAddAudio, readVideoGetDetections
    from helpers import applyFrameMemory
    from effects.blur import BlurAnnotator
    from ultralytics import YOLO
    from args import getDummyArgs

    tqdm_ = partial(tqdm, leave=False, tk_parent=app)
    for inputPath in inputFiles:
        text_field.insert("end", f"reading video {path.basename(inputPath)}\n")

        if not path.isfile(inputPath):
            text_field.insert("end",
                              f"input file {inputPath} does not exist\n")
            continue

        if inputPath.endswith("_blurred.mp4"):
            text_field.insert(
                "end", f"input file {inputPath} is a blurred file, skipping\n")
            continue

        inputName = path.splitext(path.basename(inputPath))[0]
        inputPathWithoutSrc = path.dirname(
            path.relpath(inputPath, input_entry.get()))

        outputPath = path.join(output_entry.get(), inputPathWithoutSrc,
                               f"{inputName}_blurred.mp4")

        makedirs(path.dirname(outputPath), exist_ok=True)

        if path.isfile(outputPath):
            text_field.insert(
                "end",
                f"output file {path.basename(outputPath)} already exists, skipping\n"
            )
            continue

        model = YOLO(model=model_entry.get())
        detections = readVideoGetDetections(inputPath,
                                            model,
                                            getDummyArgs(),
                                            tqdm=tqdm_)

        detections = applyFrameMemory(detections, 4)

        blurAndWriteFrames(detections,
                           BlurAnnotator(),
                           inputPath,
                           outputPath,
                           tqdm=tqdm_)

        extractAndAddAudio(inputPath,
                           outputPath,
                           log=lambda x: text_field.insert("end", x + "\n"))
    text_field.insert("end", f"finished processing videos\n", "green")

    # Define the tag and its properties after inserting the text



start_button = ct.CTkButton(input_frame, text="Start", command=runThread)
start_button.grid(row=4, column=1, padx=10, pady=10)
app.mainloop()
