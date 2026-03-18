# yolov8-learning

python -m venv .venv or python3 -m venv .venv 
source .venv/bin/activate

pip install -r requirements.txt

make sure videos are in the directory as entry_file.mp4 and exit_file.mp4

edit sim_threshold to lower number for more matches (may not be completely accurate) or higher for hopefully better accuracy

run yolo.py. takes a long time to run but works well. will hopefully save sample entry and sample exit which show what the bounding boxes look like.