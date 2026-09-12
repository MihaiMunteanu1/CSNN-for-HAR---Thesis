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
*   `include/dataset/VideoKTH_3D.h` — the dataset class (header-only). It loads the `.npy` produced by the preprocessing stage together with its `.json` sidecar, exposes the clips as 3D (T x H x W) inputs, keeps the per-frame bounding boxes attached to each sample, and selects the train / val / test subset according to `CSNN_EVAL_SPLIT`.
*   `src/sampler/HOGSampler3D.cpp` (+ `src/sampler/RandomSampler3D.cpp` as baseline) — the skeleton-based (HOG-guided) sampler. Instead of drawing patch locations uniformly over the frame, it restricts sampling to the person region given by the bounding boxes, so the filters are learned on the moving subject rather than on the static background. The generic uniform sampler is used as the baseline for comparison (`CSNN_SAMPLER=hog|random`).
*   `src/layer/ConvolutionSampler3D.cpp` — the 3D convolution layer driven by the sampler above, which learns its filters with STDP on the sampled spatio-temporal patches.

### 3. Building the KTH binaries

Each file in `apps/` becomes a target named after it. All the scripts below expect the build directory to be `cmake-build-release/` at the repository root (override with `CSNN_BUILD_DIR` if yours is named differently), so from inside it:

```bash
cmake --build . --target KTH_1layer -j$(nproc)
cmake --build . --target KTH_2layer -j$(nproc)
cmake --build . --target KTH_3layer -j$(nproc)
```

The binaries are configured entirely through environment variables (`CSNN_DATA`, `CSNN_EVAL_SPLIT`, `CSNN_SAMPLER`, `CSNN_SEED`, `CSNN_T_OBJ`, `CSNN_FH/FW/FT`, `CSNN_NF`, `CSNN_POOL_SX/SY/ST`, `CSNN_EPOCHS`, `CSNN_VIDEO_FRAMES`, ...), which is what lets the Optuna drivers launch many configurations in parallel without recompiling. Running a binary by hand is possible but not the intended way; use the scripts below.

### 4. Hyperparameter search with Optuna

The scripts in `optuna/` drive the C++ binaries: a Python driver samples a configuration, launches the binary once per seed with `CSNN_EVAL_SPLIT=val`, parses the accuracy from stdout and returns the mean across seeds as the objective. Studies are stored in a SQLite database (`optuna_studies/csnn_kth.db`) under a name that encodes depth and sampler, so a study can be resumed by relaunching the same command.

The wrappers are launched from the repository root and already default to the dataset extracted above (`../hog/kth_fullframes_tvt_19_f10_g2_runfix_80x60.npy`, i.e. a `hog/` folder next to the repository) and to the `cmake-build-release/` binaries, so they take a single argument - the sampler:

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

### 5. Final test protocol

Validation is used only to choose a configuration; the test split is touched only here. The protocol takes one configuration, re-runs it on the **test** split (`CSNN_EVAL_SPLIT=test`) with 10 seeds, prints the mean and standard deviation, and appends a row to a CSV with the full per-seed accuracies.

```bash
./optuna/run_test_1layer.sh          # 1 layer, skeleton-based sampler
./optuna/run_test_2layer.sh hog   # 2 layers
./optuna/run_test_3layer.sh hog   # 3 layers
./optuna/run_test_2layer.sh random
```

The configuration is read from the corresponding Optuna study (`optuna_studies/csnn_kth.db`), so the tuning must have run first; `STUDY_NAME` selects which study to take it from, `OUT_CSV` where to write, `N_JOBS` how many seeds run at once, and `DATA` / `INPUT_ROOT` / `CSNN_BUILD_DIR` behave as in the tuning step:

```bash
STUDY_NAME=csnn_1layer_hog OUT_CSV=data/test_1layer_hog.csv ./optuna/run_test_1layer.sh
```

Alternatively the Python drivers accept a configuration directly instead of a study, which is how a single fixed architecture is re-tested without touching Optuna:

```bash
python3 -u optuna/run_test_protocol.py \
    --binary cmake-build-release/KTH_1layer --cwd . \
    --data ../hog/kth_fullframes_tvt_19_f10_g2_runfix_80x60.npy \
    --sampler hog --n_seeds 10 \
    --t_obj 0.75 --filter_h 3 --filter_w 3 --filter_t 3 --num_filters 16 \
    --pool_sx 2 --pool_sy 2 --pool_st 1 --epochs 100 \
    --n_jobs 10 --threads_per_run 1 \
    --out_csv data/test_1layer_hog.csv
```

`run_test_protocol_2layer.py` and `run_test_protocol_3layer.py` work the same way, with `--t_obj2` / `--t_obj3` for the layer being added and `--num_filters1/2/3` for the widths. Temporal pooling must stay disabled (`--pool_st 1`, spatial-only pooling) for the multi-layer runs, otherwise the temporal dimension is exhausted at depth.

### File map

C++ side (described in section 2): `apps/kth/`, `include/dataset/VideoKTH_3D.h`, `src/sampler/HOGSampler3D.cpp`, `src/layer/ConvolutionSampler3D.cpp`.

Python side:

*   `dataset/KTH_Formatter.py` — splits the raw KTH videos into train/val/test by subject id.
*   `src/tool/extract_bboxes_kth.py` — person bounding boxes (HOG + MOG2) and clip selection.
*   `src/tool/extract_full_frames_kth.py` — the final `.npy` frames plus the `.json` metadata with bboxes and splits.
*   `optuna/` — the tuning drivers (`tune_csnn_optuna*.py`) and the test-protocol drivers (`run_test_protocol*.py`), with the shell wrappers around them.
