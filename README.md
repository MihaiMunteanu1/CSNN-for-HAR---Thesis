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

### 2. My contributions to the simulator

The KTH / HAR functionality is built on top of the existing simulator through the following files, which are the ones I created or modified:

*   `apps/kth/` — the experiment entry points: `KTH_1layer.cpp`, `KTH_2layer.cpp` and `KTH_3layer.cpp`, one per network depth. Each builds the network (input layer, 3D convolution with STDP, pooling, SVM head), reads its whole configuration from `CSNN_*` environment variables, and prints the final `classification rate: XX.XX%` line that the Optuna drivers parse.
*   `src/dataset/VideoKTH3D.cpp` — the dataset class. It loads the `.npy` produced by the preprocessing stage together with its `.json` sidecar, exposes the clips as 3D (T x H x W) inputs, keeps the per-frame bounding boxes attached to each sample, and selects the train / val / test subset according to `CSNN_EVAL_SPLIT`.
*   `src/sampler/HOGSampler3D.cpp` — the skeleton-based (HOG-guided) sampler. Instead of drawing patch locations uniformly over the frame, it restricts sampling to the person region given by the bounding boxes, so the filters are learned on the moving subject rather than on the static background. The generic uniform sampler is used as the baseline for comparison (`CSNN_SAMPLER=hog|random`).
*   `src/layer/ConvolutionSampler3D.cpp` — the 3D convolution layer driven by the sampler above, which learns its filters with STDP on the sampled spatio-temporal patches.

### 3. Building the KTH binaries

Each file in `apps/` becomes a target named after it, so from the build directory:

```bash
cmake --build . --target KTH_1layer -j$(nproc)
cmake --build . --target KTH_2layer -j$(nproc)
cmake --build . --target KTH_3layer -j$(nproc)
```

The binaries are configured entirely through environment variables (`CSNN_DATA`, `CSNN_EVAL_SPLIT`, `CSNN_SAMPLER`, `CSNN_SEED`, `CSNN_T_OBJ`, `CSNN_FH/FW/FT`, `CSNN_NF`, `CSNN_POOL_SX/SY/ST`, `CSNN_EPOCHS`, `CSNN_VIDEO_FRAMES`, ...), which is what lets the Optuna drivers launch many configurations in parallel without recompiling. Running a binary by hand is possible but not the intended way; use the scripts below.

### 4. Hyperparameter search with Optuna

The scripts in `optuna/` drive the C++ binaries: a Python driver samples a configuration, launches the binary once per seed with `CSNN_EVAL_SPLIT=val`, parses the accuracy from stdout and returns the mean across seeds as the objective. Studies are stored in a SQLite database (`optuna_studies/csnn_kth.db`) under a name that encodes depth and sampler, so a study can be resumed by relaunching the same command.

```bash
# 1-layer, skeleton-based sampling / generic sampling
DATA=../hog/kth_fullframes_tvt_19_f10_g2_runfix_80x60.npy ./optuna/run_tuning.sh hog
DATA=../hog/kth_fullframes_tvt_19_f10_g2_runfix_80x60.npy ./optuna/run_tuning.sh random

# deeper variants (conv1 / conv2 frozen at the best values found before)
./optuna/run_tuning_2layer.sh hog
./optuna/run_tuning_3layer.sh hog
```

The wrappers launch the study in a detached `tmux` session and log to `data/logs/`, so the run survives an SSH disconnect (`tmux attach -t optuna_hog` to follow it). Useful overrides, all read from the environment: `DATA` (the `.npy` from the preprocessing stage), `CSNN_BUILD_DIR`, `N_TRIALS` (default 50), `N_SEEDS` (5 for validation), `N_JOBS` (parallel runs, 32 on one Grid'5000 node), `EPOCHS`, `POOL_ST` and `STUDY_SUFFIX` (to keep several studies of the same kind apart). Note that the built-in `DATA` defaults point at older extractions, so set it explicitly.

The search space of the 1-layer study is `t_obj` in 0.40..0.80 (step 0.05), filter size in {3, 5, 7, 9} and temporal depth in {2, 3}; the deeper studies freeze the earlier layers and only tune the threshold of the newly added one.

### 5. Final test protocol

Once a study has finished, the test protocol re-runs the best configuration on the **test** split (`CSNN_EVAL_SPLIT=test`) with 10 seeds and writes one CSV with the per-seed accuracies plus mean and standard deviation. Validation is used only for choosing the configuration; the test split is touched only here.

```bash
# best config read straight from the Optuna study
STUDY_NAME=csnn_1layer_hog DATA=../hog/kth_fullframes_tvt_19_f10_g2_runfix_80x60.npy \
    ./optuna/run_test_hog.sh
./optuna/run_test_2layer.sh hog
./optuna/run_test_3layer.sh random
```

The reported results come from the scripts in `optuna/others/`, which pin the configuration explicitly instead of reading it from a study, and run the whole depth grid in one go for a given sampler:

```bash
./optuna/others/run_test_hog_runfix.sh        # 1, 2 and 3 layers, HOG sampler
./optuna/others/run_test_random_runfix.sh     # same, generic sampler
./optuna/others/run_test_hog_runfix.sh 1      # only the 1-layer run
```

Both write their CSVs to `data/test_{1,2,3}layer_{hog,random}_<tag>.csv`. Temporal pooling must stay disabled (`POOL_ST=1`, spatial-only pooling) for the multi-layer runs, otherwise the temporal dimension is exhausted at depth.

### File map

C++ side (described in section 2): `apps/kth/`, `src/dataset/VideoKTH3D.cpp`, `src/sampler/HOGSampler3D.cpp`, `src/layer/ConvolutionSampler3D.cpp`.

Python side:

*   `dataset/KTH_Formatter.py` — splits the raw KTH videos into train/val/test by subject id.
*   `src/tool/extract_bboxes_kth.py` — person bounding boxes (HOG + MOG2) and clip selection.
*   `src/tool/extract_full_frames_kth.py` — the final `.npy` frames plus the `.json` metadata with bboxes and splits.
*   `optuna/` — the tuning and test-protocol drivers, with the shell wrappers around them; `optuna/others/` holds the fixed-configuration runs that produced the reported results.
