from ultralytics/ultralytics:latest
workdir /app

# Install dependencies
copy src/ src/
copy *.pt .
copy requirements.txt .
# seems to take super long
#run conda config --add channels conda-forge && conda update -y ffmpeg
run pip install -r requirements.txt
run mkdir /app/data
CMD bash && cd /app