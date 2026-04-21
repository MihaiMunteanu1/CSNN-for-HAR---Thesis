#ifndef _ANALYSIS_SVM_H
#define _ANALYSIS_SVM_H

#include "Analysis.h"
#include "dep/libsvm/svm.h"
#include <filesystem>
#include "tool/Operations.h"

	/**
 	* @brief SVM (support vector machine), this CSNN simulator uses STDP unsupervised learning, 
	* so a classification layer is needed to evaluate the accuracy of the network.
	* Any other supervised learning method can be used, but the SVM was chosen for it's simplicity and efficacity.
    * @param draw A flag that draws the features that will be classified by the SVM, the information is recorded in the folder in the build file.
 	*/

namespace analysis {
	class Svm : public TwoPassAnalysis {

	public:
		Svm();
		Svm(const size_t &draw);

		Svm(const Svm& that) = delete;
		Svm& operator=(const Svm& that) = delete;

		virtual void resize(const Shape& shape);
		virtual void compute(const std::string& label, const Tensor<float>& sample);
		virtual void process_train(const std::string& label, const Tensor<float>& sample);
		virtual void process_test(const std::string& label, const Tensor<float>& sample);

		virtual void before_train();
		virtual void after_train();
		virtual void before_test();
		virtual void after_test();

	protected:
		// Protected templated constructor so that derived classes (e.g.
		// SvmQualitative) can register themselves in the AnalysisFactory
		// with their own RegisterClassParameter while reusing all the
		// initialization logic of the base Svm class.
		template<typename T, typename Factory>
		Svm(const RegisterClassParameter<T, Factory>& registration) :
			TwoPassAnalysis(registration),
			_c(0), _label_index(), _size(0), _node_count(0), _sample_count(0), _draw(0),
			_problem(), _model(nullptr), _train_nodes(nullptr), _test_nodes(nullptr),
			_correct_sample(0), _total_sample(0)
		{
			add_parameter("c", _c, 1.0f);
			_problem.l = 0;
			_problem.x = nullptr;
			_problem.y = nullptr;
		}

		float _c;

		std::map<std::string, double> _label_index;
		size_t _size;
		size_t _node_count;
		size_t _sample_count;
		size_t _draw;


		svm_problem _problem;
		svm_model* _model;
		svm_node* _train_nodes;
		svm_node* _test_nodes;

		size_t _correct_sample;
		size_t _total_sample;
	};
}

#endif
