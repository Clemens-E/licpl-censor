import subprocess
from args import getParsedArgs

args = getParsedArgs()

# do glob on args.input
import glob
import os
files = glob.glob(args.input)
print(f"found {len(files)} files")
for file in files:
    if (file.find("_blurred") != -1):
        print(f"skipping {file} because it already seems blurred")
        continue
    print(f"processing {file}")
    subArgs = ["python", "clean.py", "--input", file, "--batch", str(args.batch), "--frame-memory", str(args.frame_memory), "--kernel-size", str(args.kernel_size)]
    if (args.no_blur):
        subArgs.append("--no-blur")
    subprocess.run(subArgs)