# CSNN-simulator

This is a new version of the CSNN simulator that contains 2D and 3D convolution, along with two-stream methods for video analysis.

In order to run a 2D experiment, use the Convolution class in the layer, or Convolution3D while setting the temporal depth to 1.

In order to run a 3D experiment, use the Convolution3D class and set the temporal depth > 1.

In order to run a two-stream experiment, chech the TwoStream.cpp class where two experiments are created, after that, the results of these experiments are fused and evaluated using the SVM.

The SVM.cpp class can be used to test the classification rate of the SVM alone without an SNN. This is useful to make sure that the SNN is indeed adding a benefit.

The FeatureEvaluation.cpp class can be used to re-evaluate extracted and saved features by the SVM alone without re-training and re-running an SNN.

For execution policies, SparseIntermediateExecutionNew can be used for saving certain things like output features adn output timestamps (features but as spikes). If there is no need to save anything use SparseIntermediateExecution instead, it's faster. DenseIntermediateExecution is even faster.


Simulator of Convolutional Spiking Neural Network

Provide implementation of experiments described in:
* __Unsupervised Visual Feature Learning with Spike-timing-dependent Plasticity: How Far are we from Traditional Feature Learning Approaches?__, P Falez, P Tirilly, IM Bilasco, P Devienne, P Boulet, Pattern Recognition.
* __Multi-layered Spiking Neural Network with Target Timestamp Threshold Adaptation and STDP__, P Falez, P Tirilly, IM Bilasco, P Devienne, P Boulet, IJCNN 2019.

* C++ compiler (version >= 17)
* Cmake (version >= 3.1)
* Qt4 (version >= 4.4.3)
* BLAS
* LAPACKE
* OpenCV (version >= 4.2.0)

## Installation

### Dependencies

For a standard Ubuntu environment, you can install the required dependencies using the following commands:
```bash
sudo apt update
sudo apt install --yes gcc g++ make cmake libatlas-base-dev libblas-dev libopenblas-dev liblapack-dev liblapacke-dev libopencv-dev python3-opencv
sudo add-apt-repository ppa:rock-core/qt4 && sudo apt install qt4-default
```

For Grid'5000 environments, use the following commands:
```bash
sudo-g5k apt-get update
sudo-g5k apt-get install -y libopencv-dev liblapacke-dev liblapack-dev libblas-dev libopenblas-dev
```

### Compile
```bash
mkdir csnn-simulator-build
cd csnn-simulator-build
cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DUSE_GUI=NO
make -j$(nproc)
```

## Usage
Run MNIST Example:
```
export INPUT_PATH=/path/to/mnist/
./Mnist
```

## HAR on KTH Dataset

For running Human Action Recognition (HAR) on the KTH dataset, you need to follow these steps:

### 1. Preprocessing

The preprocessing stage involves extracting bounding boxes for the person in each frame and then extracting the frames themselves into a format suitable for the simulator.

First, you need to extract the bounding boxes from the KTH dataset videos. This script uses a combination of HOG and MOG2 detectors to identify the person in each frame and saves the bounding box information to a JSON file. Run the following command from the project root:

```bash
python3 src/tool/extract_bboxes_kth.py
```

Next, extract the full frames from the videos. This script reads the JSON file generated in the previous step and creates a `.npy` file containing the video frames and a corresponding `.json` file with metadata, including the bounding boxes for each frame.

```bash
python3 src/tool/extract_full_frames_kth.py
```

### 2. Running Experiments with Optuna

Once the data is preprocessed, you can run the hyperparameter optimization experiments using Optuna. The specific commands to run the Optuna experiments will depend on your experiment setup. Please refer to the relevant experiment files for detailed instructions on how to launch the Optuna studies.
