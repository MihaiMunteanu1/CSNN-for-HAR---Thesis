#ifndef _ANALYSIS_SVM_QUALITATIVE_H
#define _ANALYSIS_SVM_QUALITATIVE_H

#include "analysis/Svm.h"
#include "dataset/VideoKTH_3D.h"

#include <string>
#include <vector>
#include <map>

namespace analysis {

	/**
	 * @brief SvmQualitative extends the base Svm analysis with per-sample
	 * prediction tracking, a confusion matrix and a JSON export that links
	 * each test prediction back to its source video.
	 *
	 * This class reuses the full SVM training/inference pipeline from Svm and
	 * only overrides the test-phase hooks to record what was predicted for
	 * every sample. After the test pass it writes:
	 *
	 *   - a formatted confusion matrix to the experiment log,
	 *   - a JSON file (<experiment output>/qualitative_results_<layer>.json)
	 *     listing every test sample with its true label, predicted label and
	 *     the original video_key + group_idx obtained from
	 *     VideoKTH_3D::get_test_sample_mapping().
	 *
	 * The JSON file is meant to be consumed by the companion Python script
	 * src/tool/visualize_qualitative.py which extracts the actual frames from
	 * the video files and organises them by correct / misclassified outcomes.
	 *
	 * @note Relies on VideoKTH_3D passing the action name as the sample label
	 *       (e.g. "boxing", "running") rather than a numeric index. See the
	 *       corresponding change in src/dataset/VideoKTH_3D.cpp.
	 */
	class SvmQualitative : public Svm {

	public:
		SvmQualitative();

		SvmQualitative(const SvmQualitative& that) = delete;
		SvmQualitative& operator=(const SvmQualitative& that) = delete;

		// Only the test-phase hooks are overridden. All the training logic
		// (compute, process_train, before_train, after_train) is inherited
		// unchanged from Svm.
		virtual void before_test() override;
		virtual void process_test(const std::string& label, const Tensor<float>& sample) override;
		virtual void after_test() override;

	private:
		struct SampleRecord {
			size_t sample_idx;
			std::string true_label;
			std::string predicted_label;
			std::string video_key;
			size_t group_idx;
			bool correct;
		};

		// All predictions made during the test pass, in sample order.
		std::vector<SampleRecord> _records;

		// confusion_matrix[true_label][predicted_label] = count
		std::map<std::string, std::map<std::string, size_t>> _confusion_matrix;

		// Sorted list of all label names seen so far (used for consistent
		// matrix formatting and JSON serialization).
		std::vector<std::string> _class_names;

		// Helper: build _class_names from _label_index (inherited).
		void build_class_names();

		// Helper: format the confusion matrix as a text block and write it
		// to the experiment log. Rows = true label, columns = predicted.
		void print_confusion_matrix();

		// Helper: write the JSON file with per-sample results.
		void save_json();
	};

}

#endif
