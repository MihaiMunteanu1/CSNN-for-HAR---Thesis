#ifndef _DATASET_VIDEO_KTH_3D_H
#define _DATASET_VIDEO_KTH_3D_H

#include <filesystem>
#include <string>
#include <cassert>
#include <fstream>
#include <limits>
#include <tuple>
#include <math.h>
#include <vector>
#include <map>

#include "Tensor.h"
#include "Input.h"
#include "tool/Operations.h"

#define VIDEO_KTH3D_WIDTH 1
#define VIDEO_KTH3D_HEIGHT 1
#define VIDEO_KTH3D_DEPTH 1
#define VIDEO_KTH3D_CONV_DEPTH 1

namespace dataset
{
	/**
	 * @brief VideoKTH_3D loads KTH videos using pre-computed HOG frame data.
	 *
	 * Instead of random frame sampling, this class reads a JSON file produced by
	 * extract_person_frames.py which contains the specific frame indices and
	 * bounding boxes where persons are detected.
	 *
	 * Each video sample extracts frames from one pre-computed temporal group,
	 * guaranteeing that the person is visible in the selected frames.
	 */
	class VideoKTH_3D : public Input
	{

	public:
		struct FrameBBox {
			int frame_idx;
			std::vector<std::tuple<int, int, int, int>> bboxes; // (x, y, w, h)
		};

		struct VideoGroup {
			std::vector<FrameBBox> frames;
		};

		struct VideoHOGData {
			int total_video_frames;
			std::vector<VideoGroup> groups;
		};

		VideoKTH_3D(const std::string &video_folder_name, const std::string &hog_json_path,
					 const size_t &frame_per_video, const size_t &frame_gap = 0, const size_t &threshold = 0,
					 const size_t &sample_per_video = 0, const size_t &grey_video = 0,
					 std::string exp_name = "", const size_t &draw = 0,
					 const size_t &frame_size_width = 0, const size_t &frame_size_height = 0,
					 size_t max_read = std::numeric_limits<size_t>::max());

		virtual bool has_next() const;
		virtual std::pair<std::string, Tensor<InputType>> next();
		virtual uint32_t assign_label_to_sample(std::string _current_video_name);
		virtual void set_frame_gap(int _frame_gap, cv::VideoCapture& capture, cv::Mat& skipFrame);
		virtual bool movement_threshold(cv::Mat frame, cv::Mat next_frame);
		virtual cv::Mat frame_preprocess(int _frame_preprocess, cv::Mat frame, cv::Mat next_frame);
		virtual void save_as_images(std::pair<std::string, Tensor<InputType>> out);

		virtual void reset();
		virtual void close();

		size_t size() const;
		virtual std::string to_string() const;
		virtual const Shape &shape() const;

		/**
		 * @brief Returns the bounding boxes for a given sample index and temporal frame.
		 * Used by HOG_Convolution3D to get pre-computed bounding boxes.
		 */
		static const std::map<std::string, VideoHOGData>& get_hog_data();

		/**
		 * @brief Returns the sample mapping built during data loading.
		 * Maps sample_index -> (video_key, group_idx) for correct HOG lookup.
		 */
		static const std::map<size_t, std::pair<std::string, size_t>>& get_train_sample_mapping();
		static const std::map<size_t, std::pair<std::string, size_t>>& get_test_sample_mapping();

		static void reset_sample_mappings();

		/**
		 * @brief Returns true if the loaded NPY metadata contains per-sample
		 * bboxes (full-frame mode). False for legacy crop mode.
		 */
		static bool has_bboxes();

		/**
		 * @brief Returns (x, y, w, h) for the bbox of train sample `local_idx`
		 * at temporal frame `t`. Coordinates are in output-resolution pixel
		 * space (matches the NPY frame layout). Returns (0,0,0,0) if no bbox.
		 */
		static std::tuple<int, int, int, int> get_train_sample_bbox(size_t local_idx, size_t t);
		static std::tuple<int, int, int, int> get_val_sample_bbox(size_t local_idx, size_t t);
		static std::tuple<int, int, int, int> get_test_sample_bbox(size_t local_idx, size_t t);

		static const std::map<size_t, std::pair<std::string, size_t>>& get_val_sample_mapping();


	private:
		void load_hog_json(const std::string &json_path);
		std::string get_relative_video_key(const std::string &video_path) const;

		// NPY cache mode (raw uint8 frames pre-extracted by cnn_har_app/extract_frames_kth.py)
		void load_npy_cache(const std::string &npy_path);
		std::pair<std::string, Tensor<InputType>> next_from_cache();

		uint32_t swap(uint32_t v);

		std::string _video_folder_path;
		uint32_t _frame_size_width;
		uint32_t _frame_size_height;

		std::vector<std::string> _video_list;
		std::vector<std::string> _action_list;

		uint32_t _cursor;
		uint32_t _cursor_count;
		uint32_t _sample_per_video;

		uint32_t _size;
		uint32_t _frame_number;
		uint32_t _label_count;

		Shape _shape;

		std::string _current_video_name;
		std::string _video_name_buffer;
		int _frame_per_video;
		uint32_t _grey_video;
		int _frame_gap;
		int _frame_gap_counter;
		int _frame_preprocess;
		std::string _file_path;

		int _draw;
		std::string _exp_name;
		int _threshold;

		uint32_t _max_read;

		// HOG pre-computed data
		std::string _hog_json_path;
		static std::map<std::string, VideoHOGData> _hog_data;
		static bool _hog_data_loaded;

		// Sample index -> (video_key, group_idx) mapping, built during next()
		static std::map<size_t, std::pair<std::string, size_t>> _train_sample_mapping;
		static std::map<size_t, std::pair<std::string, size_t>> _val_sample_mapping;
		static std::map<size_t, std::pair<std::string, size_t>> _test_sample_mapping;
		static size_t _train_sample_counter;
		static size_t _val_sample_counter;
		static size_t _test_sample_counter;
		// Active split for this dataset instance: "train", "val", or "test".
		// Inferred from _video_folder_path in the constructor.
		std::string _split_name;

		// NPY cache: pre-extracted person-cropped frames, loaded once.
		// Layout: _npy_frames is flat, sample i offset = i * T * H * W.
		bool _use_npy_cache;
		static bool _npy_loaded;
		static std::vector<uint8_t> _npy_frames;
		static size_t _npy_N, _npy_T, _npy_H, _npy_W;
		static std::vector<int> _npy_labels;
		static std::vector<std::string> _npy_actions;
		static std::vector<std::string> _npy_video_keys;
		static std::vector<size_t> _npy_group_idx;
		static std::vector<std::string> _npy_splits;
		// Per-instance: indices into the global cache filtered by this split.
		std::vector<size_t> _split_sample_indices;

		// Per-sample bboxes from NPY metadata. Flat layout: bbox(i, t, c) =
		// _npy_bboxes[((i * T) + t) * 4 + c] with c in {0:x, 1:y, 2:w, 3:h}.
		// Empty when the metadata does not provide bboxes (legacy crop mode).
		static bool _npy_has_bboxes;
		static std::vector<int> _npy_bboxes;

		// Global indices of train/val/test samples in the same order they will
		// be emitted by next(). Built once from _npy_splits so that static
		// accessors (used by HOGSampler3D) can map per-split local idx ->
		// global idx.
		static std::vector<size_t> _train_global_indices;
		static std::vector<size_t> _val_global_indices;
		static std::vector<size_t> _test_global_indices;
	};

}

#endif