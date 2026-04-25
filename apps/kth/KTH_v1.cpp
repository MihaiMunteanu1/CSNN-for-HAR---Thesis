#include "Experiment.h"
#include "dataset/VideoKTH_3D.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "stdp/Proportional.h"
#include "layer/Convolution3D.h"
#include "Distribution.h"
#include "execution/DenseIntermediateExecution.h"
#include "execution/SparseIntermediateExecutionNew.h"
#include "analysis/Svm.h"
#include "analysis/SvmQualitative.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "layer/Pooling.h"
#include "process/OnOffFilter.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "process/MaxScaling.h"
#include "stdp/Linear.h"
#include "stdp/BiologicalMultiplicative.h"
#include "analysis/SaveOutput.h"
#include "sampler/RandomSampler3D.h"
#include "sampler/HOGSampler3D.h"
#include "layer/ConvolutionSampler3D.h"
#include "dataset/VideoKTH_3D.h"
#include "dataset/ImageSequenceKTH.h"
#include "process/SpikingBackgroundSubtraction.h"
//numactl --interleave=all ./KTH_v1


///TODO
///incercat cu stride 2

int main(int argc, char **argv)
{
    int seed = 42; //42,7,123
    Experiment<SparseIntermediateExecutionNew> experiment(
            argv, argc,
            "result_vv1/seed_" + std::to_string(seed),
            "model_vv1/seed_" + std::to_string(seed),
            "kth_vv1_" + std::to_string(seed),
            seed, true,
            false, false
    );

    size_t frame_size_width = 80;
    size_t frame_size_height = 60;
    size_t video_frames = 7; // kernel size
    size_t frame_gap = 0; //skip
    size_t grey = 1;
    size_t threshold = 5;
    size_t train_sample_per_video = 10;
    size_t test_sample_per_video = 10;
    size_t draw = 0;

    size_t tmp_filter_size = 2;
    size_t temp_stride = 1; // 1/2

    experiment.push<process::DefaultOnOffFilter>(7, 1.0, 4.0);
    experiment.push<process::MaxScaling>();
    experiment.push<LatencyCoding>();
//    experiment.push<process::SpikingBackgroundSubtraction>(experiment.name(), 1, 0);


    ///export INPUT_PATH="/mnt/c/Users/**path_to**/kth_organized/"
//    const char *input_path_ptr = std::getenv("INPUT_PATH");
//    if (input_path_ptr == nullptr)
//    {
//        throw std::runtime_error("Require to define kth dataset organized input path");
//    }
//    std::string input_path(input_path_ptr);

//    std::string input_path = "/home/mihai/kth_organized/";
    std::string input_path = "/home/mmuntean/kth_organized/";

    std::string hog_json_path = "../hog/hog_person_data_" + std::to_string(video_frames) + ".json";

    std::cout<<hog_json_path<<std::endl;

    dataset::VideoKTH_3D::reset_sample_mappings();

    experiment.add_train<dataset::VideoKTH_3D>(
            input_path + "train/", hog_json_path, video_frames, frame_gap, threshold,
            train_sample_per_video, grey, experiment.name(), draw,
            frame_size_width, frame_size_height);
    experiment.add_test<dataset::VideoKTH_3D>(
            input_path + "test/", hog_json_path, video_frames, frame_gap, threshold,
            test_sample_per_video, grey, experiment.name(), draw,
            frame_size_width, frame_size_height);


    float th_lr = 1.0f;
    float w_lr = 0.1f; //0.1

    float t_obj1 = 0.75f;
    float t_obj2 = 0.75f;
    float t_obj3 = 0.75f;

    std::string weights_path =  "Weights/kth_vv1_123_2/3/kth_vv1_123_2.json";
    auto &conv1 = experiment.push<layer::ConvolutionSampler3D>(96, 5, 5, 2, "", 1, 1, 1);
    conv1.set_name("conv1");
    conv1.parameter<bool>("draw").set(false);
    conv1.parameter<bool>("save_weights").set(false);
    conv1.parameter<bool>("save_random_start").set(false);
    conv1.parameter<bool>("log_spiking_neuron").set(false);
    conv1.parameter<bool>("inhibition").set(true);
    conv1.parameter<uint32_t>("epoch").set(100); //150
    conv1.parameter<float>("annealing").set(0.95f);
    conv1.parameter<float>("min_th").set(1.0f);
    conv1.parameter<float>("t_obj").set(t_obj1);
    conv1.parameter<float>("lr_th").set(th_lr);
    conv1.parameter<bool>("wta_infer").set(false);
    conv1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
    conv1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(8.0, 0.1);
    conv1.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1f);
    conv1.parameter<Sampler>("sampler").set<sampler::HOGSampler3D>(
            static_cast<size_t>(1), static_cast<size_t>(1));

    auto &pool1 = experiment.push<layer::Pooling3D>(2, 2, 1, 2, 2, 1);
    pool1.set_name("pool1");

    std::string weights_path_conv2 =  "Weights/kth_vv1_123_2/5/kth_vv1_123_2.json";
    auto &conv2 = experiment.push<layer::ConvolutionSampler3D>(96, 5, 5, tmp_filter_size, "", 1, 1, temp_stride);
    conv2.set_name("conv2");
    conv2.parameter<bool>("draw").set(false);
    conv2.parameter<bool>("save_weights").set(true);
    conv2.parameter<bool>("save_random_start").set(false);
    conv2.parameter<bool>("log_spiking_neuron").set(false);
    conv2.parameter<bool>("inhibition").set(true);
    conv2.parameter<uint32_t>("epoch").set(100);
    conv2.parameter<float>("annealing").set(0.97f);
    conv2.parameter<float>("min_th").set(1.0f);
    conv2.parameter<size_t>("max_train_spikes").set(3000);
    conv2.parameter<float>("t_obj").set(t_obj2);
    conv2.parameter<float>("lr_th").set(th_lr);
    conv2.parameter<bool>("wta_infer").set(false);
    conv2.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
    conv2.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(30.0, 2.0);
    conv2.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1f);
    conv2.parameter<Sampler>("sampler").set<sampler::HOGSampler3D>(
            static_cast<size_t>(2), static_cast<size_t>(2));

    auto &pool2 = experiment.push<layer::Pooling3D>(2, 2, 1, 2, 2, 1);
    pool2.set_name("pool2");

//    auto &fc1 = experiment.push<layer::ConvolutionSampler3D>(128, 12, 17, tmp_filter_size, "", 1, 1, temp_stride);
    auto &pool2b = experiment.push<layer::Pooling3D>(2, 2, 1, 2, 2, 1);
    pool2b.set_name("pool2b");
    auto &fc1 = experiment.push<layer::ConvolutionSampler3D>(128, 6, 8, tmp_filter_size, "", 1, 1, temp_stride);
    fc1.set_name("fc1");

    fc1.parameter<bool>("draw").set(false);
    fc1.parameter<bool>("save_weights").set(true);
    fc1.parameter<bool>("save_random_start").set(false);
    fc1.parameter<bool>("log_spiking_neuron").set(false);
    fc1.parameter<bool>("inhibition").set(true);
    fc1.parameter<uint32_t>("epoch").set(50);
    fc1.parameter<float>("annealing").set(0.97f);
    fc1.parameter<float>("min_th").set(1.0f);
    fc1.parameter<size_t>("max_train_spikes").set(2000);
    fc1.parameter<float>("t_obj").set(t_obj3);
    fc1.parameter<float>("lr_th").set(th_lr);
    fc1.parameter<bool>("wta_infer").set(true);
    fc1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
    fc1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(15.0, 0.1);
    fc1.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1f);
    fc1.parameter<Sampler>("sampler").set<sampler::HOGSampler3D>(
            static_cast<size_t>(4), static_cast<size_t>(4));

//#ifdef ENABLE_QT
//    conv1.plot_threshold(true);
//	conv1.plot_reconstruction(true);
//#endif

//    // --- conv1 output ---
//    auto &conv1_save = experiment.output<TimeObjectiveOutput>(conv1, t_obj1);
//    conv1_save.add_analysis<analysis::SaveOutput>("conv1_train", "conv1_test", false);

    auto &conv1_out = experiment.output<TimeObjectiveOutput>(conv1, t_obj1);
    conv1_out.add_postprocessing<process::SumPooling>(2,2);
    conv1_out.add_postprocessing<process::FeatureScaling>();
    conv1_out.add_analysis<analysis::Activity>();
    conv1_out.add_analysis<analysis::Coherence>();
    conv1_out.add_analysis<analysis::Svm>();

//    // --- conv2 output ---
    auto &conv2_save = experiment.output<TimeObjectiveOutput>(conv2, t_obj2);
    conv2_save.add_analysis<analysis::SaveOutput>("conv2_train", "conv2_test", false);

    auto &conv2_out = experiment.output<TimeObjectiveOutput>(conv2, t_obj2);
    conv2_out.add_postprocessing<process::SumPooling>(2, 2);
    conv2_out.add_postprocessing<process::FeatureScaling>();
    conv2_out.add_analysis<analysis::Activity>();
    conv2_out.add_analysis<analysis::Coherence>();
    conv2_out.add_analysis<analysis::Svm>();

//    // --- fc1 output ---
    auto &fc1_save = experiment.output<TimeObjectiveOutput>(fc1, t_obj3);
    fc1_save.add_analysis<analysis::SaveOutput>("fc1_train", "fc1_test", false);

    auto &fc1_out = experiment.output<TimeObjectiveOutput>(fc1, t_obj3);
    fc1_out.add_postprocessing<process::FeatureScaling>();
    fc1_out.add_analysis<analysis::Activity>();
    fc1_out.template add_analysis<analysis::SvmQualitative>();


    experiment.run(10000);

    return experiment.wait();
}