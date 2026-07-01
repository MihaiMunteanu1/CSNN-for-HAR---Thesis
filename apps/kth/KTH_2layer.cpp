// KTH 2-layer CSNN + SVM head, fully parameterized via env vars.
//
// Architecture: conv1 -> pool1 -> conv2 -> pool2 -> SVM(conv2_out)
//
// Conv1 reuses CSNN_FH/FW/FT/NF/T_OBJ — the best params from the 1-layer
// OPTUNA study should be fixed there (HOG 9x9x3/t_obj=0.80, Random 5x5x3/t_obj=0.75).
// Conv2 takes its own t_obj2 via CSNN_T_OBJ2 (the only OPTUNA-tuned param);
// kernel shape mirrors conv1 (same FH/FW/FT) and number of filters defaults to 32.
//
// Env vars (defaults in parentheses):
//   CSNN_SEED          (42)
//   CSNN_T_OBJ         (0.75)   conv1 t_obj
//   CSNN_T_OBJ2        (0.75)   conv2 t_obj — the OPTUNA target
//   CSNN_FH/FW/FT      (3/3/2)  shared by conv1 AND conv2
//   CSNN_NF            (16)     conv1 filters
//   CSNN_NF2           (32)     conv2 filters
//   CSNN_POOL_SX/SY/ST (2/2/2)  shared by pool1 AND pool2
//   CSNN_SAMPLER       (hog)    hog|random — applied to both conv layers
//   CSNN_EVAL_SPLIT    (val)    val|test
//   CSNN_DATA          (../hog/kth_fullframes_tvt_g4_80x60.npy)
//   CSNN_INPUT_ROOT    (/home/mmuntean/kth_organized_tvt/)
//   CSNN_EPOCHS        (100)    conv1 epochs
//   CSNN_EPOCHS2       (80)     conv2 epochs (KTH_v1.cpp:159 used 80)
//   CSNN_TH_MEAN       (5.0)    conv1 threshold init mean
//   CSNN_TH_DEV        (1.0)    conv1 threshold init std
//   CSNN_TH_MEAN2      (5.0)    conv2 threshold init mean
//   CSNN_TH_DEV2       (1.0)    conv2 threshold init std
//   CSNN_W_LR          (0.1)    STDP weight LR for conv1
//   CSNN_W_LR2         (0.05)   STDP weight LR for conv2 (half, per KTH_v1.cpp:168)
//   CSNN_TAG           ("")

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

#include "Experiment.h"
#include "dataset/VideoKTH_3D.h"
#include "stdp/Biological.h"
#include "layer/Convolution3D.h"
#include "layer/ConvolutionSampler3D.h"
#include "layer/Pooling.h"
#include "Distribution.h"
#include "execution/SparseIntermediateExecutionNew.h"
#include "analysis/Svm.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "process/OnOffFilter.h"
#include "process/MaxScaling.h"
#include "process/Pooling.h"
#include "process/Scaling.h"
#include "sampler/RandomSampler3D.h"
#include "sampler/HOGSampler3D.h"

static std::string env_str(const char *name, const std::string &defv) {
	const char *v = std::getenv(name);
	return (v && *v) ? std::string(v) : defv;
}

static int env_int(const char *name, int defv) {
	const char *v = std::getenv(name);
	return (v && *v) ? std::atoi(v) : defv;
}

static float env_float(const char *name, float defv) {
	const char *v = std::getenv(name);
	return (v && *v) ? static_cast<float>(std::atof(v)) : defv;
}


int main(int argc, char **argv)
{
	int seed                 = env_int("CSNN_SEED", 42);
	float t_obj1             = env_float("CSNN_T_OBJ", 0.75f);
	float t_obj2             = env_float("CSNN_T_OBJ2", 0.75f);
	int filter_h             = env_int("CSNN_FH", 3);
	int filter_w             = env_int("CSNN_FW", 3);
	int filter_t             = env_int("CSNN_FT", 2);
	int num_filters1         = env_int("CSNN_NF", 16);
	int num_filters2         = env_int("CSNN_NF2", 32);
	int pool_sx              = env_int("CSNN_POOL_SX", 2);
	int pool_sy              = env_int("CSNN_POOL_SY", 2);
	int pool_st              = env_int("CSNN_POOL_ST", 2);
	std::string sampler_kind = env_str("CSNN_SAMPLER", "hog");
	std::string eval_split   = env_str("CSNN_EVAL_SPLIT", "val");
	std::string data_path    = env_str("CSNN_DATA", "../hog/kth_fullframes_tvt_g4_80x60.npy");
	std::string input_root   = env_str("CSNN_INPUT_ROOT", "/home/mmuntean/kth_organized_tvt/");
	int epochs1              = env_int("CSNN_EPOCHS", 100);
	int epochs2              = env_int("CSNN_EPOCHS2", 80);
	float th_mean1           = env_float("CSNN_TH_MEAN", 5.0f);
	float th_dev1            = env_float("CSNN_TH_DEV", 1.0f);
	float th_mean2           = env_float("CSNN_TH_MEAN2", 5.0f);
	float th_dev2            = env_float("CSNN_TH_DEV2", 1.0f);
	float w_lr1              = env_float("CSNN_W_LR", 0.1f);
	float w_lr2              = env_float("CSNN_W_LR2", 0.05f);
	std::string tag          = env_str("CSNN_TAG", "");

	if (eval_split != "val" && eval_split != "test") {
		std::cerr << "ERROR: CSNN_EVAL_SPLIT must be 'val' or 'test', got '"
				  << eval_split << "'" << std::endl;
		return 1;
	}
	if (sampler_kind != "hog" && sampler_kind != "random") {
		std::cerr << "ERROR: CSNN_SAMPLER must be 'hog' or 'random', got '"
				  << sampler_kind << "'" << std::endl;
		return 1;
	}

	std::string suffix = "_s" + std::to_string(seed)
						+ "_t1_" + std::to_string(t_obj1)
						+ "_t2_" + std::to_string(t_obj2)
						+ "_f" + std::to_string(filter_h) + "x" + std::to_string(filter_w)
						+ "x" + std::to_string(filter_t);
	if (!tag.empty()) suffix += "_" + tag;

	std::filesystem::create_directories("result_2layer/" + eval_split + suffix);
	std::filesystem::create_directories("model_2layer/" + eval_split + suffix);

	Experiment<SparseIntermediateExecutionNew> experiment(
		argv, argc,
		"result_2layer/" + eval_split + suffix,
		"model_2layer/" + eval_split + suffix,
		"kth_2layer" + suffix,
		seed, true,
		false, false
	);

	std::cout << "=== KTH_2layer config ===" << std::endl;
	std::cout << "  seed         = " << seed << std::endl;
	std::cout << "  t_obj1       = " << t_obj1 << std::endl;
	std::cout << "  t_obj2       = " << t_obj2 << std::endl;
	std::cout << "  filter       = " << filter_h << "x" << filter_w << "x" << filter_t
			  << " (shared conv1/conv2)" << std::endl;
	std::cout << "  num_filters  = " << num_filters1 << " (conv1) / "
			  << num_filters2 << " (conv2)" << std::endl;
	std::cout << "  pool         = " << pool_sx << "x" << pool_sy << "x" << pool_st
			  << " (shared pool1/pool2)" << std::endl;
	std::cout << "  sampler      = " << sampler_kind << std::endl;
	std::cout << "  eval_split   = " << eval_split << std::endl;
	std::cout << "  data         = " << data_path << std::endl;
	std::cout << "  input_root   = " << input_root << std::endl;
	std::cout << "  epochs       = " << epochs1 << " (conv1) / " << epochs2 << " (conv2)" << std::endl;
	std::cout << "  th_init1     = N(" << th_mean1 << ", " << th_dev1 << ")" << std::endl;
	std::cout << "  th_init2     = N(" << th_mean2 << ", " << th_dev2 << ")" << std::endl;
	std::cout << "  w_lr         = " << w_lr1 << " (conv1) / " << w_lr2 << " (conv2)" << std::endl;
	std::cout << "=========================" << std::endl;

	size_t frame_size_width  = 0;
	size_t frame_size_height = 0;

	size_t video_frames = static_cast<size_t>(env_int("CSNN_VIDEO_FRAMES", 19));
	size_t frame_gap = 2;
	size_t grey = 1;
	size_t threshold = 5;
	size_t train_sample_per_video = 5;
	size_t test_sample_per_video = 5;
	size_t draw = 0;

	dataset::VideoKTH_3D::reset_sample_mappings();

	experiment.add_train<dataset::VideoKTH_3D>(
		input_root + "train/", data_path, video_frames, frame_gap, threshold,
		train_sample_per_video, grey, experiment.name(), draw,
		frame_size_width, frame_size_height);
	experiment.add_test<dataset::VideoKTH_3D>(
		input_root + eval_split + "/", data_path, video_frames, frame_gap, threshold,
		test_sample_per_video, grey, experiment.name(), draw,
		frame_size_width, frame_size_height);

	experiment.push<process::DefaultOnOffFilter>(7, 1.0, 4.0);
	experiment.push<process::MaxScaling>();
	experiment.push<LatencyCoding>();

	float th_lr = 1.0f;

	auto &conv1 = experiment.push<layer::ConvolutionSampler3D>(
		num_filters1, filter_h, filter_w, filter_t, "", 1, 1, 2);
	conv1.set_name("conv1");
	conv1.parameter<bool>("draw").set(false);
	conv1.parameter<bool>("save_weights").set(false);
	conv1.parameter<bool>("save_random_start").set(false);
	conv1.parameter<bool>("log_spiking_neuron").set(false);
	conv1.parameter<bool>("inhibition").set(true);
	conv1.parameter<uint32_t>("epoch").set(static_cast<uint32_t>(epochs1));
	conv1.parameter<float>("annealing").set(0.97f);
	conv1.parameter<float>("min_th").set(1.0f);
	conv1.parameter<float>("t_obj").set(t_obj1);
	conv1.parameter<float>("lr_th").set(th_lr);
	conv1.parameter<size_t>("max_train_spikes").set(0);
	conv1.parameter<bool>("wta_infer").set(false);
	conv1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
	conv1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(th_mean1, th_dev1);
	conv1.parameter<STDP>("stdp").set<stdp::Biological>(w_lr1, 0.1f);

	if (sampler_kind == "hog")
		conv1.parameter<Sampler>("sampler").set<sampler::HOGSampler3D>(1, 1);
	else
		conv1.parameter<Sampler>("sampler").set<sampler::RandomSampler3D>();

	auto &pool1 = experiment.push<layer::Pooling3D>(
		pool_sx, pool_sy, 1, pool_sx, pool_sy, 1);
	pool1.set_name("pool1");

	auto &conv2 = experiment.push<layer::ConvolutionSampler3D>(
		num_filters2, filter_h, filter_w, filter_t, "", 1, 1, 2);
	conv2.set_name("conv2");
	conv2.parameter<bool>("draw").set(false);
	conv2.parameter<bool>("save_weights").set(false);
	conv2.parameter<bool>("save_random_start").set(false);
	conv2.parameter<bool>("log_spiking_neuron").set(false);
	conv2.parameter<bool>("inhibition").set(true);
	conv2.parameter<uint32_t>("epoch").set(static_cast<uint32_t>(epochs2));
	conv2.parameter<float>("annealing").set(0.95f);
	conv2.parameter<float>("min_th").set(1.0f);
	conv2.parameter<float>("t_obj").set(t_obj2);
	conv2.parameter<float>("lr_th").set(th_lr);
	conv2.parameter<size_t>("max_train_spikes").set(0);
	conv2.parameter<bool>("wta_infer").set(false);
	conv2.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
	conv2.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(th_mean2, th_dev2);
	conv2.parameter<STDP>("stdp").set<stdp::Biological>(w_lr2, 0.1f);

	if (sampler_kind == "hog")
		conv2.parameter<Sampler>("sampler").set<sampler::HOGSampler3D>(2, 2);
	else
		conv2.parameter<Sampler>("sampler").set<sampler::RandomSampler3D>();

	auto &pool2 = experiment.push<layer::Pooling3D>(
		pool_sx, pool_sy, 1, pool_sx, pool_sy, 1);
	pool2.set_name("pool2");

	auto &conv2_out = experiment.output<TimeObjectiveOutput>(conv2, t_obj2);
	conv2_out.add_postprocessing<process::SumPooling>(2, 2);
	conv2_out.add_postprocessing<process::FeatureScaling>();
	conv2_out.add_analysis<analysis::Activity>();
	conv2_out.add_analysis<analysis::Coherence>();
	conv2_out.add_analysis<analysis::Svm>();

	experiment.run(10000);
	int rc = experiment.wait();

	std::cout << "RESULT"
			  << " seed=" << seed
			  << " t_obj1=" << t_obj1
			  << " t_obj2=" << t_obj2
			  << " fh=" << filter_h
			  << " fw=" << filter_w
			  << " ft=" << filter_t
			  << " nf1=" << num_filters1
			  << " nf2=" << num_filters2
			  << " pool=" << pool_sx << "x" << pool_sy << "x" << pool_st
			  << " sampler=" << sampler_kind
			  << " eval=" << eval_split
			  << " rc=" << rc
			  << std::endl;

	return rc;
}