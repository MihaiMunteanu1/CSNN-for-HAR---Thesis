# CSNN-simulator

For Grid'5000 environments:
```bash
For GRID-5000 server:
mkdir cmake-build-release
cd cmake-build-release
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

Finally, extract the full frames from the videos. 
This script reads the JSON file generated in the previous step and 
creates a `.npy` file containing the video frames and a corresponding 
`.json` file with metadata, including the bounding boxes for each frame.

```bash
python3 src/tool/extract_full_frames_kth.py \
    --bbox_json hog/hog_person_data_tvt_19_f10_g2_runfix.json \
    --video_root /path/to/kth_split \
    --frame_size_width 80 --frame_size_height 60 \
    --output hog/kth_fullframes_tvt_19_f10_g2_runfix_80x60
```

The frames are stored at 80x60 (the resolution used by the CSNN experiments) 
while the bboxes are rescaled from the 160x120 detection resolution to 
match. The resulting `.npy` is what the Optuna scripts are pointed
at through their `DATA` variable.

### 2. Contributions to the simulator

*   `apps/kth/` — the experiment entry points: `KTH_1layer.cpp`, `KTH_2layer.cpp` and `KTH_3layer.cpp`
*   `include/dataset/VideoKTH_3D.h` — It loads the `.npy` produced by the preprocessing stage together with its `.json` sidecar, exposes the clips as 3D (T x H x W) inputs, keeps the per-frame bounding boxes attached to each sample
*   `src/sampler/HOGSampler3D.cpp` (`src/sampler/RandomSampler3D.cpp` as baseline) — the skeleton-based (HOG-guided) sampler. Instead of drawing patch locations uniformly over the frame, it restricts sampling to the person region given by the bounding boxes, so the filters are learned on the moving subject rather than on the static background. 
*   `src/layer/ConvolutionSampler3D.cpp` — the 3D convolution layer driven by the sampler above, which learns its filters with STDP on the sampled spatio-temporal patches.

### 3. KTH binaries


```bash
cmake --build . --target KTH_1layer -j$(nproc)
cmake --build . --target KTH_2layer -j$(nproc)
cmake --build . --target KTH_3layer -j$(nproc)
```


### 4. Hyperparameter search with Optuna

The Python scripts in `optuna/` optimize the C++ binaries by running sampled configurations across multiple seeds (CSNN_EVAL_SPLIT=val) and averaging the output accuracies. Studies are saved in optuna_studies/csnn_kth.db and can be easily resumed.

Launch the wrappers from the repository root with a single argument: the sampler. Paths to the dataset (../hog/) and binaries (cmake-build-release/) are already configured by default.
```bash
./optuna/run_tuning_1layer.sh hog          # 1 layer, skeleton-based sampling
./optuna/run_tuning_1layer.sh random       # 1 layer, generic sampling (baseline)
./optuna/run_tuning_2layer.sh hog   # conv1 frozen at the 1-layer best, tunes conv2
./optuna/run_tuning_3layer.sh hog   # conv1+conv2 frozen, tunes conv3
```

They launch the study in a detached `tmux` session and log to `data/logs/`, so the run survives an SSH disconnect (`tmux attach -t optuna_hog` to follow it). Everything else is overridable through the environment, without touching the scripts: `DATA` (the `.npy` from the preprocessing stage; the path is resolved from the repository root, so use `DATA=hog/....npy` if you kept it inside the repo), `INPUT_ROOT` (the split folder from step 1), `CSNN_BUILD_DIR`, `N_TRIALS` (default 50), `N_SEEDS` (5 for validation), `N_JOBS` (parallel runs, 32 on one Grid'5000 node), `EPOCHS` (`EPOCHS1`/`EPOCHS2`/`EPOCHS3` for the deeper studies), `POOL_ST` and `STUDY_SUFFIX` (to keep several studies of the same kind apart), e.g.:

```bash
N_TRIALS=100 N_JOBS=32 STUDY_SUFFIX=v2 ./optuna/run_tuning_1layer.sh hog
```

The search space of the 1-layer study is `t_obj` in 0.40..0.80 (step 0.05), filter size in {3, 5, 7, 9} and temporal depth in {2, 3}; the deeper studies freeze the earlier layers and only tune the threshold of the newly added one.

### 5. Test protocol

Validation is used only to choose a configuration; the test split is touched only here. The protocol takes one configuration, re-runs it on the **test** split (`CSNN_EVAL_SPLIT=test`) with 10 seeds, prints the mean and standard deviation, and appends a row to a CSV with the full per-seed accuracies.

```bash
./optuna/run_test_1layer.sh hog   # 1 layer, skeleton-based sampler
./optuna/run_test_2layer.sh hog   # 2 layers
./optuna/run_test_3layer.sh hog   # 3 layers
./optuna/run_test_2layer.sh random
```

The configuration is read from the corresponding Optuna study (`optuna_studies/csnn_kth.db`), so the tuning must have run first. The study name defaults to the one the tuning step creates (`csnn_1layer_hog`, `csnn_2layer_random`, ...); `STUDY_NAME` overrides it, which is what a study launched with a `STUDY_SUFFIX` needs. `OUT_CSV` chooses where the results are written, `N_JOBS` how many seeds run at once, and `DATA` / `INPUT_ROOT` / `CSNN_BUILD_DIR` behave as in the tuning step:

```bash
STUDY_NAME=csnn_1layer_hog_19f OUT_CSV=data/test_1layer_hog.csv ./optuna/run_test_1layer.sh hog
```

Each wrapper is a thin layer over the corresponding Python driver (`run_test_protocol_1layer.py`, `run_test_protocol_2layer.py`, `run_test_protocol_3layer.py`), which can also be called directly with an explicit configuration instead of a study - `--t_obj` (or `--t_obj2` / `--t_obj3` for the layer being added), `--filter_h/w/t`, `--num_filters1/2/3`, `--pool_sx/sy/st`, `--epochs*`. Temporal pooling must stay disabled (`--pool_st 1`, i.e. `POOL_ST=1`, spatial-only pooling) for the multi-layer runs, otherwise the temporal dimension is exhausted at depth.

### File map

C++ side (described in section 2): `apps/kth/`, `include/dataset/VideoKTH_3D.h`, `src/sampler/HOGSampler3D.cpp`, `src/layer/ConvolutionSampler3D.cpp`.

Python side:

*   `dataset/KTH_Formatter.py` — splits the raw KTH videos into train/val/test by subject id.
*   `src/tool/extract_bboxes_kth.py` — person bounding boxes (HOG + MOG2) and clip selection.
*   `src/tool/extract_full_frames_kth.py` — the final `.npy` frames plus the `.json` metadata with bboxes and splits.
*   `optuna/` — the tuning drivers (`tune_csnn_optuna*.py`) and the test-protocol drivers (`run_test_protocol*.py`), with the shell wrappers around them.
