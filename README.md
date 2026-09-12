# CSNN-simulator

For Grid'5000 environments:
```bash
For GRID-5000 server:
mkdir csnn-simulator-build
cd csnn-simulator-build
cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DUSE_GUI=NO
make -j$(nproc)
sudo-g5k apt-get update
sudo-g5k apt-get install -y libopencv-dev liblapacke-dev liblapack-dev libblas-dev libopenblas-dev
```


## HAR on KTH Dataset

### 1. Preprocessing

Reorganise the raw KTH videos into a train / validation / test split.
```bash
python3 dataset/KTH_Formatter.py --input /path/to/kth_raw --output /path/to/kth_split
```

Next, you need to extract the bounding boxes from the KTH dataset videos. This script uses a combination of HOG and MOG2 detectors to identify the person in each frame and saves the bounding box information to a JSON file. `--input_path` must point to the split produced at the previous step (the folder that contains `train/`, `val/`, `test/`), and `--output` is the JSON that the next step will read:

```bash
python3 src/tool/extract_bboxes_kth.py \
    --input_path /path/to/kth_split/ \
    --temporal_kernel 19 --frame_gap 2 --num_groups 10 \
    --frame_width 160 --frame_height 120 \
    --output hog/hog_person_data_tvt_19_f10_g2.json
```

These are the settings used for the reported experiments: clips of T = 19 frames taken every 2nd frame, 10 clips per video, detection done at 160x120. If `--input_path` / `--output` are omitted the script falls back to hardcoded paths (`/home/mmuntean/kth_organized_tvt/`), so always pass your own.

The `running` class is the hardest one for the detector (the person leaves the frame, and a clip is kept only if all 19 frames have a valid bbox). It was re-extracted with more permissive thresholds and merged back into the JSON above, which gives the `runfix` JSON actually used:

```bash
python3 src/tool/extract_bboxes_kth.py \
    --input_path /path/to/kth_split/ \
    --temporal_kernel 19 --frame_gap 2 --num_groups 10 \
    --frame_width 160 --frame_height 120 \
    --only_action running \
    --hit_threshold -1.2 --min_bbox_area_ratio 0.004 \
    --min_bbox_aspect 0.15 --max_bbox_aspect 2.0 \
    --mog2_min_area 300 --max_carry 4 \
    --merge_into hog/hog_person_data_tvt_19_f10_g2.json \
    --output    hog/hog_person_data_tvt_19_f10_g2_runfix.json
```

Finally, extract the full frames from the videos. This script reads the JSON file generated in the previous step and creates a `.npy` file containing the video frames and a corresponding `.json` file with metadata, including the bounding boxes for each frame. `--video_root` must be the same split folder as above (its default points elsewhere), `--bbox_json` the JSON from the previous step, and `--output` a path *without* extension - `.npy` and `.json` are appended:

```bash
python3 src/tool/extract_full_frames_kth.py \
    --bbox_json hog/hog_person_data_tvt_19_f10_g2_runfix.json \
    --video_root /path/to/kth_split \
    --frame_size_width 80 --frame_size_height 60 \
    --output hog/kth_fullframes_tvt_19_f10_g2_runfix_80x60
```

The frames are stored at 80x60 (the resolution used by the CSNN experiments) while the bboxes are rescaled from the 160x120 detection resolution to match. The resulting `.npy` is what the Optuna scripts are pointed at through their `DATA` variable; the `.json` sidecar must stay next to it, with the same name.

### 2. Running Experiments with Optuna

Once the data is preprocessed, you can run the hyperparameter optimization experiments using Optuna. The specific commands to run the Optuna experiments will depend on your experiment setup. Please refer to the relevant experiment files for detailed instructions on how to launch the Optuna studies. The shell wrappers in `optuna/` read the dataset path from a `DATA` environment variable, e.g. `DATA=../hog/kth_fullframes_tvt_19_f10_g2_runfix_80x60.npy ./optuna/run_test_hog.sh` (their built-in default points at an older file, so set it explicitly).

### Relevant Files for KTH HAR

The following files represent my contributions to the project for the Human Action Recognition functionality on the KTH dataset:

*   `apps/kth/`: This directory contains the main application files for running the KTH experiments.
*   `src/dataset/VideoKTH3D.cpp`: This class is responsible for loading and managing the KTH video data.
*   `dataset/KTH_Formatter.py`: Splits the raw KTH videos into train/val/test folders by subject id.
*   `src/sampler/HOGSampler3D.cpp`: This sampler is used to extract 3D patches from the video data, guided by the HOG detector's bounding boxes.
*   `src/layer/ConvolutionSampler3D.cpp`: This layer performs 3D convolution on the sampled data.
*   `optuna/`: This directory contains the scripts and configurations for running hyperparameter optimization with Optuna.
*   `src/tool/extract_bboxes_kth.py`: The script for extracting person bounding boxes from the KTH videos.
*   `src/tool/extract_full_frames_kth.py`: The script for extracting full frames and generating the final `.npy` and `.json` files.
