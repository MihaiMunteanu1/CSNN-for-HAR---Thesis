#include "analysis/SvmQualitative.h"

#include "Experiment.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>


/***
 */

using namespace analysis;

static RegisterClassParameter<SvmQualitative, AnalysisFactory> _svm_qual_register("SvmQualitative");

SvmQualitative::SvmQualitative() :
	Svm(_svm_qual_register),
	_records(),
	_confusion_matrix(),
	_class_names()
{
}

void SvmQualitative::before_test() {
	Svm::before_test();

	_records.clear();
	_confusion_matrix.clear();
	_class_names.clear();
}

void SvmQualitative::process_test(const std::string& label, const Tensor<float>& sample) {
	// Mirrors Svm::process_test but captures the predicted label so we can
	// build per-sample records and a confusion matrix.
	size_t node_cursor = 0;
	for (size_t j = 0; j < _size; j++) {
		float v = sample.at_index(j);
		if (v != 0.0f) {
			_test_nodes[node_cursor].index = static_cast<int>(j + 1);
			_test_nodes[node_cursor].value = v;
			node_cursor++;
		}
	}
	_test_nodes[node_cursor].index = -1;

	double y_pred = ::svm_predict(_model, _test_nodes);

	// Resolve the predicted numeric index back to a class name. Labels were
	// inserted in insertion order into _label_index during the compute pass.
	std::string predicted_label;
	for (const auto& kv : _label_index) {
		if (kv.second == y_pred) {
			predicted_label = kv.first;
			break;
		}
	}
	if (predicted_label.empty()) {
		predicted_label = "<unknown>";
	}

	auto it = _label_index.find(label);
	bool correct = (it != std::end(_label_index) && y_pred == it->second);
	if (correct) {
		_correct_sample++;
	}

	SampleRecord rec;
	rec.sample_idx = _total_sample;
	rec.true_label = label;
	rec.predicted_label = predicted_label;
	rec.video_key = "";   // filled in after_test() from VideoKTH_3D mapping
	rec.group_idx = 0;
	rec.correct = correct;
	_records.push_back(rec);

	_confusion_matrix[label][predicted_label]++;

	_total_sample++;
}

void SvmQualitative::build_class_names() {
	_class_names.clear();
	for (const auto& kv : _label_index) {
		_class_names.push_back(kv.first);
	}
	std::sort(_class_names.begin(), _class_names.end());
}

void SvmQualitative::print_confusion_matrix() {
	if (_class_names.empty()) {
		return;
	}

	size_t col_w = 6;
	for (const auto& name : _class_names) {
		col_w = std::max(col_w, name.size() + 1);
	}

	std::ostringstream oss;
	oss << "===Confusion Matrix (rows = true, cols = predicted)===" << std::endl;

	// Header row
	oss << std::setw(static_cast<int>(col_w)) << "true\\pred";
	for (const auto& c : _class_names) {
		oss << std::setw(static_cast<int>(col_w)) << c;
	}
	oss << std::endl;

	for (const auto& true_name : _class_names) {
		oss << std::setw(static_cast<int>(col_w)) << true_name;
		for (const auto& pred_name : _class_names) {
			size_t count = 0;
			auto row_it = _confusion_matrix.find(true_name);
			if (row_it != _confusion_matrix.end()) {
				auto col_it = row_it->second.find(pred_name);
				if (col_it != row_it->second.end()) {
					count = col_it->second;
				}
			}
			oss << std::setw(static_cast<int>(col_w)) << count;
		}
		oss << std::endl;
	}

	experiment().log() << oss.str() << std::endl;
}


static std::string json_escape(const std::string& s) {
	std::string out;
	out.reserve(s.size() + 2);
	for (char c : s) {
		switch (c) {
			case '"':  out += "\\\""; break;
			case '\\': out += "\\\\"; break;
			case '\n': out += "\\n";  break;
			case '\r': out += "\\r";  break;
			case '\t': out += "\\t";  break;
			default:
				if (static_cast<unsigned char>(c) < 0x20) {
					char buf[8];
					snprintf(buf, sizeof(buf), "\\u%04x", c);
					out += buf;
				} else {
					out += c;
				}
				break;
		}
	}
	return out;
}

void SvmQualitative::save_json() {
	const auto& mapping = dataset::VideoKTH_3D::get_test_sample_mapping();
	for (auto& rec : _records) {
		auto it = mapping.find(rec.sample_idx);
		if (it != mapping.end()) {
			rec.video_key = it->second.first;
			rec.group_idx = it->second.second;
		}
	}

	// Compose output path: <experiment output>/qualitative_results_L<layer>.json
	std::string dir = experiment().output_path();
	if (dir.empty()) {
		dir = ".";
	}
	std::filesystem::create_directories(dir);

	std::string path = dir + "/qualitative_results_L" +
		std::to_string(layer_index()) + ".json";

	std::ofstream f(path);
	if (!f.is_open()) {
		experiment().log() << "SvmQualitative: cannot open " << path
			<< " for writing" << std::endl;
		return;
	}

	double accuracy = (_total_sample == 0) ? 0.0 :
		(static_cast<double>(_correct_sample) / static_cast<double>(_total_sample));

	f << "{\n";
	f << "  \"experiment\": \"" << json_escape(experiment().name()) << "\",\n";
	f << "  \"layer_index\": " << layer_index() << ",\n";
	f << "  \"accuracy\": " << std::fixed << std::setprecision(6) << accuracy << ",\n";
	f << "  \"correct\": " << _correct_sample << ",\n";
	f << "  \"total\": " << _total_sample << ",\n";

	// Classes array
	f << "  \"classes\": [";
	for (size_t i = 0; i < _class_names.size(); i++) {
		if (i > 0) f << ", ";
		f << "\"" << json_escape(_class_names[i]) << "\"";
	}
	f << "],\n";


	f << "  \"confusion_matrix\": [\n";
	for (size_t i = 0; i < _class_names.size(); i++) {
		f << "    [";
		for (size_t j = 0; j < _class_names.size(); j++) {
			if (j > 0) f << ", ";
			size_t count = 0;
			auto row_it = _confusion_matrix.find(_class_names[i]);
			if (row_it != _confusion_matrix.end()) {
				auto col_it = row_it->second.find(_class_names[j]);
				if (col_it != row_it->second.end()) {
					count = col_it->second;
				}
			}
			f << count;
		}
		f << "]";
		if (i + 1 < _class_names.size()) f << ",";
		f << "\n";
	}
	f << "  ],\n";


	f << "  \"samples\": [\n";
	for (size_t i = 0; i < _records.size(); i++) {
		const auto& r = _records[i];
		f << "    {";
		f << "\"sample_idx\": " << r.sample_idx << ", ";
		f << "\"true_label\": \"" << json_escape(r.true_label) << "\", ";
		f << "\"predicted_label\": \"" << json_escape(r.predicted_label) << "\", ";
		f << "\"video_key\": \"" << json_escape(r.video_key) << "\", ";
		f << "\"group_idx\": " << r.group_idx << ", ";
		f << "\"correct\": " << (r.correct ? "true" : "false");
		f << "}";
		if (i + 1 < _records.size()) f << ",";
		f << "\n";
	}
	f << "  ]\n";
	f << "}\n";

	f.close();

	experiment().log() << "SvmQualitative: wrote " << path << std::endl;
}

void SvmQualitative::after_test() {
	build_class_names();
	print_confusion_matrix();
	save_json();

	Svm::after_test();
}
