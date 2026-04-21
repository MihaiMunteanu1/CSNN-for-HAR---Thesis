#include "analysis/Svm.h"

#include "Experiment.h"

using namespace analysis;

static RegisterClassParameter<Svm, AnalysisFactory> _register("Svm");

Svm::Svm() : TwoPassAnalysis(_register),
			 _c(0), _label_index(), _size(0), _node_count(0), _sample_count(0),
			 _problem(), _model(nullptr), _train_nodes(nullptr), _test_nodes(nullptr),
			 _correct_sample(0), _total_sample(0)
{

	add_parameter("c", _c, 1.0f);

	_problem.l = 0;
	_problem.x = nullptr;
	_problem.y = nullptr;
}

Svm::Svm(const size_t &draw) : TwoPassAnalysis(_register),
							   _draw(draw), _c(0), _label_index(), _size(0), _node_count(0), _sample_count(0),
							   _problem(), _model(nullptr), _train_nodes(nullptr), _test_nodes(nullptr),
							   _correct_sample(0), _total_sample(0)
{

	add_parameter("c", _c, 10.0f); //1.0f

	_problem.l = 0;
	_problem.x = nullptr;
	_problem.y = nullptr;
}

void Svm::resize(const Shape& shape) {
	_node_count = 0;
	_sample_count = 0;
	_size = shape.product();
	_label_index.clear();
}

void Svm::compute(const std::string& label, const Tensor<float>& sample) {
	if(_label_index.find(label) == std::end(_label_index)) {
		_label_index.emplace(label, _label_index.size());
	}

	for(size_t j=0; j<_size; j++) {
		if(sample.at_index(j) != 0.0) {
			_node_count++;
		}
	}
	_node_count++;
	_sample_count++;

	//draw_progress(_sample_count, get_train_count());

	if (_draw == 1)
	{
		std::string _file_path = std::filesystem::current_path();
		std::string _expName = experiment().name();
		//TODO: find layer index.
		std::string _LayerIndex = std::to_string(0);
		std::filesystem::create_directories(_file_path + "/ExtractedFeatures/SVM/" + _expName + "_" + _LayerIndex + "/");
		SaveWeights(_file_path + "/ExtractedFeatures/SVM/" + _expName + "_" + _LayerIndex + "/" + _expName + "_" + _LayerIndex + ".json", label, sample);
		// Tensor<float>::draw_feature_tensor(_file_path + "/ExtractedFeatures/SVM/" + _expName + "_" + _LayerIndex + "/" + _expName + "_" + _LayerIndex + "_" + std::to_string(_sample_count) + "_", sample);
		Tensor<float>::draw_tensor(_file_path + "/ExtractedFeatures/SVM/" + _expName + "_" + _LayerIndex + "/" + _expName + "_" + _LayerIndex + "_" + std::to_string(_sample_count) + "_", sample);
	}
}

void Svm::before_train() {
	_train_nodes = new struct svm_node[_node_count];
	_test_nodes = new struct svm_node[_size];

	_problem.l = _sample_count;
	_problem.y = new double[_sample_count];
	_problem.x = new struct svm_node*[_sample_count];

	_sample_count = 0;
	_node_count = 0;

}

void Svm::process_train(const std::string& label, const Tensor<float>& sample) {
	_problem.y[_sample_count] = _label_index[label];
	_problem.x[_sample_count] = _train_nodes+_node_count;

	for(size_t j=0; j<_size; j++) {
		float v = sample.at_index(j);

		if(v != 0.0) {
			_train_nodes[_node_count].index = j+1;
			_train_nodes[_node_count].value = v;
			_node_count++;
		}


	}
	_train_nodes[_node_count].index = -1;
	_node_count++;

	_sample_count++;
}

void Svm::after_train() {
	struct svm_parameter parameters;
	parameters.svm_type = C_SVC;
	parameters.kernel_type = LINEAR;
	parameters.degree = 3;
	parameters.gamma = 1.0/static_cast<float>(_size);
	parameters.coef0 = 0;
	parameters.nu = 0.5;
	parameters.cache_size = 100;
	parameters.eps = 1e-3;
	parameters.p = 0.1;
	parameters.shrinking = 1;
	parameters.probability = 0;
	parameters.nr_weight = 0;
	parameters.weight_label = NULL;
	parameters.weight = NULL;

	// --- GRID SEARCH pe C cu 5-fold CV ---
	const std::vector<double> C_grid = {0.01, 0.1, 1.0, 10.0, 100.0};
	const int n_folds = 5;
	double best_C = 1.0;
	double best_acc = -1.0;

	std::vector<double> target(_problem.l);
	for (double C_try : C_grid) {
		parameters.C = C_try;
		::svm_cross_validation(&_problem, &parameters, n_folds, target.data());
		int correct = 0;
		for (int i = 0; i < _problem.l; i++)
			if (target[i] == _problem.y[i]) correct++;
		double acc = static_cast<double>(correct) / _problem.l;
		experiment().print() << "  CV C=" << C_try << " -> " << (acc*100) << "%" << std::endl;
		if (acc > best_acc) { best_acc = acc; best_C = C_try; }
	}
	experiment().print() << "Best C=" << best_C << " (CV acc " << (best_acc*100) << "%)" << std::endl;

	// Antrenare finală cu best C pe tot train set-ul
	parameters.C = best_C;
	experiment().print() << "Train svm" << std::endl;
	_model = ::svm_train(&_problem, &parameters);
}


//void Svm::after_train() {
//	struct svm_parameter parameters;
//
//	parameters.svm_type = C_SVC;
//	parameters.kernel_type = LINEAR;
//	parameters.degree = 3;
//	parameters.gamma = 1.0/static_cast<float>(_size);
//	parameters.coef0 = 0;
//	parameters.nu = 0.5;
//	parameters.cache_size = 100;
//	parameters.C = _c;
//	parameters.eps = 1e-3;
//	parameters.p = 0.1;
//	parameters.shrinking = 1;
//	parameters.probability = 0;
//	parameters.nr_weight = 0;
//	parameters.weight_label = NULL;
//	parameters.weight = NULL;
//
//	experiment().print() << "Train svm" << std::endl;
//	_model = ::svm_train(&_problem, &parameters);
//}

void Svm::before_test() {
	_correct_sample = 0;
	_total_sample = 0;

}

void Svm::process_test(const std::string& label, const Tensor<float>& sample) {
	size_t node_cursor = 0;
	for(size_t j=0; j<_size; j++) {
		float v = sample.at_index(j);

		if(v != 0.0) {
			_test_nodes[node_cursor].index = j+1;
			_test_nodes[node_cursor].value = v;
			node_cursor++;
		}
	}
	_test_nodes[node_cursor].index = -1;

	double y_pred = ::svm_predict(_model, _test_nodes);

	auto it = _label_index.find(label);

	if(it != std::end(_label_index) && y_pred == it->second) {
		_correct_sample++;
	}
	_total_sample++;
}

void Svm::after_test() {
	experiment().log() << "===SVM===" << std::endl;
	experiment().log() << "classification rate: " <<
						 (static_cast<float>(_correct_sample)/static_cast<float>(_total_sample)*100.0) << "% (" <<
						 _correct_sample << "/" << _total_sample << ")" << std::endl;
	experiment().log() << std::endl;

	delete[] _problem.y;
	_problem.y = nullptr;
	delete[] _problem.x;
	_problem.x = nullptr;
	delete[] _train_nodes;
	_train_nodes = nullptr;
	delete[] _test_nodes;
	_test_nodes = nullptr;

	svm_free_and_destroy_model(&_model);
	_model = nullptr;
}
