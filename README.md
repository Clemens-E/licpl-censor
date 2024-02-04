# Licpl-Censor
## [Deutsches Readme](README.de.md)
Licpl-Censor is a tool designed to censor faces and license plates in videos. It's currently under active development, so expect frequent updates and potential bugs.

## Limitations
The model is specifically trained on:
- German license plates. While it may work with other EU plates, results may vary.
- Dashcam videos. The model is accustomed to the perspective and quality of dashcam videos, so performance may differ with other video types.

The dataset is continuously updated and improved. However, not all camera types are well supported yet. As this is a hobby project, the process of labeling and improving the dataset is time-consuming.

Please note that the model may miss instances, requiring manual review of the output. Nonetheless, it significantly reduces the time compared to manual censoring.

## Installation
### Standalone Executable (Windows only)
There is a standalone executable available for Windows.
It includes all required dependencies and NVIDIA CUDA support.
You *might* have to install [ffmpeg](https://community.chocolatey.org/packages/ffmpeg) manually if it's not already installed on your system, or if the tool can't find it.

The standalone executable is *not* available in the releases, for the simple reason that it's too large to upload to GitHub.
Download links are attached to the release notes.

There exists a cli and a gui version, the cli version has all features, while the gui version is more user-friendly.

__Heads up:__ When opening the standalone executable, it will extract the files to a temporary directory. This might take a few seconds or even minutes, depending on your system.


### Docker

1. Clone the repository.
2. Run `docker build -t licpl-censor .` in the root directory.
3. Run the container with `docker run --rm -it -v /path/to/your/workdir:/app/data licpl-censor bash`. Make sure to pass through any NVIDIA GPU you have to avoid slow performance.

You can then run the tool with the following command:

```bash
python src/cli.py --input /app/data/your-input/ --output /app/data/your-output/ --model /app/data/model.pt
```

The tool will attempt to replicate the folder structure of your input folder in the output folder. If this is not desired or fails, you can use the `--flat--output` flag to disable this behavior.

To view all available options, run `python src/cli.py --help``.

## Getting the Model
New models will be released as they become available. These "small" models offer a good balance between speed and accuracy. Larger models are also in development but are not currently available for release.

# Donations
This project is a hobby, developed in my free time. If you'd like to support the growth and improvement of the dataset and model, you can do so here:

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/clemense)

### Acknowledgements
The model is trained with Ultralytics YOLOv8.