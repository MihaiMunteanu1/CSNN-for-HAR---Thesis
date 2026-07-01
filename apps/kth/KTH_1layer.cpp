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
#include "analysis/SvmQualitative.h"
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
	int seed                = env_int("CSNN_SEED", 42);
	float t_obj             = env_float("CSNN_T_OBJ", 0.75f);
	int filter_h            = env_int("CSNN_FH", 3);
	int filter_w            = env_int("CSNN_FW", 3);
	int filter_t            = env_int("CSNN_FT", 2);
	int num_filters         = env_int("CSNN_NF", 16);
	int pool_sx             = env_int("CSNN_POOL_SX", 2);
	int pool_sy             = env_int("CSNN_POOL_SY", 2);
	int pool_st             = env_int("CSNN_POOL_ST", 2);
	std::string sampler_kind = env_str("CSNN_SAMPLER", "hog");
	std::string eval_split   = env_str("CSNN_EVAL_SPLIT", "val");
	std::string data_path    = env_str("CSNN_DATA", "../hog/kth_fullframes_tvt_g4_80x60.npy");
	std::string input_root   = env_str("CSNN_INPUT_ROOT", "/home/mmuntean/kth_organized_tvt/");
	int epochs              = env_int("CSNN_EPOCHS", 100);
	int qualitative         = env_int("CSNN_QUALITATIVE", 0);
	float th_mean           = env_float("CSNN_TH_MEAN", 5.0f);
	float th_dev            = env_float("CSNN_TH_DEV", 1.0f);
	float w_lr              = env_float("CSNN_W_LR", 0.1f);
	std::string tag         = env_str("CSNN_TAG", "");

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

	std::string suffix = "_s" + std::to_string(seed) + "_t" + std::to_string(t_obj)
						+ "_f" + std::to_string(filter_h) + "x" + std::to_string(filter_w)
						+ "x" + std::to_string(filter_t);
	if (!tag.empty()) suffix += "_" + tag;

	std::filesystem::create_directories("result_1layer/" + eval_split + suffix);
	std::filesystem::create_directories("model_1layer/" + eval_split + suffix);

	Experiment<SparseIntermediateExecutionNew> experiment(
		argv, argc,
		"result_1layer/" + eval_split + suffix,
		"model_1layer/" + eval_split + suffix,
		"kth_1layer" + suffix,
		seed, true,
		false, false
	);

	std::cout << "=== KTH_1layer config ===" << std::endl;
	std::cout << "  seed         = " << seed << std::endl;
	std::cout << "  t_obj        = " << t_obj << std::endl;
	std::cout << "  filter       = " << filter_h << "x" << filter_w << "x" << filter_t << std::endl;
	std::cout << "  num_filters  = " << num_filters << std::endl;
	std::cout << "  pool         = " << pool_sx << "x" << pool_sy << "x" << pool_st << std::endl;
	std::cout << "  sampler      = " << sampler_kind << std::endl;
	std::cout << "  eval_split   = " << eval_split << std::endl;
	std::cout << "  data         = " << data_path << std::endl;
	std::cout << "  input_root   = " << input_root << std::endl;
	std::cout << "  epochs       = " << epochs << std::endl;
	std::cout << "  th_init      = N(" << th_mean << ", " << th_dev << ")" << std::endl;
	std::cout << "=========================" << std::endl;

	size_t frame_size_width  = 0;
	size_t frame_size_height = 0;
	size_t video_frames = static_cast<size_t>(env_int("CSNN_VIDEO_FRAMES", 19));
	size_t frame_gap = 2;
	size_t grey = 1;
	size_t threshold = 5;
	size_t train_sample_per_video = 5; //skip
	size_t test_sample_per_video = 5; //skip
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
		num_filters, filter_h, filter_w, filter_t, "", 1, 1, 2);
	conv1.set_name("conv1");
	conv1.parameter<bool>("draw").set(false);
	conv1.parameter<bool>("save_weights").set(false);
	conv1.parameter<bool>("save_random_start").set(false);
	conv1.parameter<bool>("log_spiking_neuron").set(false);
	conv1.parameter<bool>("inhibition").set(true);
	conv1.parameter<uint32_t>("epoch").set(static_cast<uint32_t>(epochs));
	conv1.parameter<float>("annealing").set(0.97f);
	conv1.parameter<float>("min_th").set(1.0f);
	conv1.parameter<float>("t_obj").set(t_obj);
	conv1.parameter<float>("lr_th").set(th_lr);
	conv1.parameter<size_t>("max_train_spikes").set(0);
	conv1.parameter<bool>("wta_infer").set(false);
	conv1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
	conv1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(th_mean, th_dev);
	conv1.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1f);

	if (sampler_kind == "hog")
		conv1.parameter<Sampler>("sampler").set<sampler::HOGSampler3D>(1, 1);
	else
		conv1.parameter<Sampler>("sampler").set<sampler::RandomSampler3D>();

	auto &pool1 = experiment.push<layer::Pooling3D>(
		pool_sx, pool_sy, pool_st, pool_sx, pool_sy, pool_st);
	pool1.set_name("pool1");

	auto &conv1_out = experiment.output<TimeObjectiveOutput>(conv1, t_obj);
	conv1_out.add_postprocessing<process::SumPooling>(2, 2);
	conv1_out.add_postprocessing<process::FeatureScaling>();
	conv1_out.add_analysis<analysis::Activity>();
	conv1_out.add_analysis<analysis::Coherence>();
	if (qualitative)
		conv1_out.add_analysis<analysis::SvmQualitative>();
	else
		conv1_out.add_analysis<analysis::Svm>();

	experiment.run(10000);
	int rc = experiment.wait();

	std::cout << "RESULT"
			  << " seed=" << seed
			  << " t_obj=" << t_obj
			  << " fh=" << filter_h
			  << " fw=" << filter_w
			  << " ft=" << filter_t
			  << " nf=" << num_filters
			  << " pool=" << pool_sx << "x" << pool_sy << "x" << pool_st
			  << " sampler=" << sampler_kind
			  << " eval=" << eval_split
			  << " rc=" << rc
			  << std::endl;

	return rc;
}
