#include "dataset/VideoKTH_3D.h"
#include <iostream>
#include <sstream>

using namespace dataset;

// Static members
std::map<std::string, VideoKTH_3D::VideoHOGData> VideoKTH_3D::_hog_data;
bool VideoKTH_3D::_hog_data_loaded = false;

/**
 * @brief Simple JSON string value parser - extracts value for a key from a JSON-like string.
 */
static std::string json_get_string(const std::string &json, const std::string &key)
{
	std::string search = "\"" + key + "\"";
	size_t pos = json.find(search);
	if (pos == std::string::npos) return "";

	pos = json.find("\"", pos + search.length() + 1);
	if (pos == std::string::npos) return "";
	pos++;

	size_t end = json.find("\"", pos);
	if (end == std::string::npos) return "";

	return json.substr(pos, end - pos);
}

static int json_get_int(const std::string &json, const std::string &key)
{
	std::string search = "\"" + key + "\"";
	size_t pos = json.find(search);
	if (pos == std::string::npos) return 0;

	pos = json.find(":", pos);
	if (pos == std::string::npos) return 0;
	pos++;

	while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t'))
		pos++;

	std::string num;
	while (pos < json.size() && (std::isdigit(json[pos]) || json[pos] == '-'))
	{
		num += json[pos];
		pos++;
	}

	return num.empty() ? 0 : std::stoi(num);
}

void VideoKTH_3D::load_hog_json(const std::string &json_path)
{
	if (_hog_data_loaded)
		return;

	std::ifstream file(json_path);
	if (!file.is_open())
	{
		std::cerr << "WARNING: Cannot open HOG JSON file: " << json_path << std::endl;
		std::cerr << "VideoKTH_3D will fall back to standard Video frame sampling." << std::endl;
		_hog_data_loaded = true;
		return;
	}

	std::stringstream buffer;
	buffer << file.rdbuf();
	std::string content = buffer.str();
	file.close();

	// Parse the JSON manually (to avoid external dependency)
	// Structure: {"config": {...}, "videos": {"key": {"total_video_frames": N, "groups": [...]}}}

	// Find "videos" section
	size_t videos_pos = content.find("\"videos\"");
	if (videos_pos == std::string::npos)
	{
		std::cerr << "WARNING: Invalid HOG JSON format (no 'videos' key)" << std::endl;
		_hog_data_loaded = true;
		return;
	}

	// Find the opening brace of videos object
	size_t videos_start = content.find("{", videos_pos + 8);
	if (videos_start == std::string::npos)
	{
		_hog_data_loaded = true;
		return;
	}

	// Parse each video entry
	size_t pos = videos_start + 1;
	int video_count = 0;

	while (pos < content.size())
	{
		// Find next video key (e.g., "train/boxing/video.avi")
		size_t key_start = content.find("\"", pos);
		if (key_start == std::string::npos || key_start >= content.size() - 1)
			break;

		key_start++;
		size_t key_end = content.find("\"", key_start);
		if (key_end == std::string::npos)
			break;

		std::string video_key = content.substr(key_start, key_end - key_start);

		// Check if this looks like a video path (contains /)
		if (video_key.find("/") == std::string::npos)
		{
			pos = key_end + 1;
			// We might have hit the closing brace of videos
			if (content[pos] == '}')
				break;
			continue;
		}

		// Find "total_video_frames"
		size_t tvf_pos = content.find("\"total_video_frames\"", key_end);
		VideoHOGData hog_data;
		hog_data.total_video_frames = 0;

		if (tvf_pos != std::string::npos)
		{
			size_t colon = content.find(":", tvf_pos + 20);
			if (colon != std::string::npos)
			{
				size_t num_start = colon + 1;
				while (num_start < content.size() && content[num_start] == ' ')
					num_start++;
				std::string num;
				while (num_start < content.size() && std::isdigit(content[num_start]))
				{
					num += content[num_start];
					num_start++;
				}
				if (!num.empty())
					hog_data.total_video_frames = std::stoi(num);
			}
		}

		// Find "groups" array
		size_t groups_pos = content.find("\"groups\"", key_end);
		if (groups_pos == std::string::npos)
		{
			pos = key_end + 1;
			continue;
		}

		size_t groups_array_start = content.find("[", groups_pos);
		if (groups_array_start == std::string::npos)
		{
			pos = key_end + 1;
			continue;
		}

		// Parse groups: each group is an array of frame objects
		// [ [ {frame_idx, bboxes}, ... ], [ ... ], ... ]
		int bracket_depth = 0;
		size_t gpos = groups_array_start;

		// Skip the outer [
		gpos++;
		bracket_depth = 1;

		while (gpos < content.size() && bracket_depth > 0)
		{
			// Skip whitespace
			while (gpos < content.size() && (content[gpos] == ' ' || content[gpos] == '\n' || content[gpos] == '\r' || content[gpos] == '\t' || content[gpos] == ','))
				gpos++;

			if (gpos >= content.size() || content[gpos] == ']')
			{
				bracket_depth--;
				gpos++;
				break;
			}

			if (content[gpos] != '[')
			{
				gpos++;
				continue;
			}

			// Start of a group array
			gpos++; // skip [
			VideoGroup group;

			while (gpos < content.size())
			{
				// Skip whitespace/commas
				while (gpos < content.size() && (content[gpos] == ' ' || content[gpos] == '\n' || content[gpos] == '\r' || content[gpos] == '\t' || content[gpos] == ','))
					gpos++;

				if (gpos >= content.size() || content[gpos] == ']')
				{
					gpos++; // skip ]
					break;
				}

				if (content[gpos] != '{')
				{
					gpos++;
					continue;
				}

				// Parse frame object: {"frame_idx": N, "bboxes": [...]}
				size_t obj_start = gpos;
				int brace_depth = 1;
				gpos++;
				while (gpos < content.size() && brace_depth > 0)
				{
					if (content[gpos] == '{')
						brace_depth++;
					else if (content[gpos] == '}')
						brace_depth--;
					gpos++;
				}

				std::string frame_obj = content.substr(obj_start, gpos - obj_start);

				FrameBBox fb;
				fb.frame_idx = json_get_int(frame_obj, "frame_idx");

				// Parse bboxes array
				size_t bboxes_pos = frame_obj.find("\"bboxes\"");
				if (bboxes_pos != std::string::npos)
				{
					size_t bb_arr_start = frame_obj.find("[", bboxes_pos);
					if (bb_arr_start != std::string::npos)
					{
						size_t bb_pos = bb_arr_start + 1;
						while (bb_pos < frame_obj.size())
						{
							while (bb_pos < frame_obj.size() && (frame_obj[bb_pos] == ' ' || frame_obj[bb_pos] == '\n' || frame_obj[bb_pos] == '\r' || frame_obj[bb_pos] == '\t' || frame_obj[bb_pos] == ','))
								bb_pos++;

							if (bb_pos >= frame_obj.size() || frame_obj[bb_pos] == ']')
								break;

							if (frame_obj[bb_pos] != '{')
							{
								bb_pos++;
								continue;
							}

							// Parse single bbox object
							size_t bbox_start = bb_pos;
							int bd = 1;
							bb_pos++;
							while (bb_pos < frame_obj.size() && bd > 0)
							{
								if (frame_obj[bb_pos] == '{')
									bd++;
								else if (frame_obj[bb_pos] == '}')
									bd--;
								bb_pos++;
							}

							std::string bbox_str = frame_obj.substr(bbox_start, bb_pos - bbox_start);
							int bx = json_get_int(bbox_str, "x");
							int by = json_get_int(bbox_str, "y");
							int bw = json_get_int(bbox_str, "w");
							int bh = json_get_int(bbox_str, "h");

							fb.bboxes.push_back(std::make_tuple(bx, by, bw, bh));
						}
					}
				}

				group.frames.push_back(fb);
			}

			if (!group.frames.empty())
			{
				hog_data.groups.push_back(group);
			}
		}

		_hog_data[video_key] = hog_data;
		video_count++;

		pos = gpos;
	}

	std::cout << "VideoKTH_3D: Loaded HOG data for " << video_count << " videos from " << json_path << std::endl;
	_hog_data_loaded = true;
}

std::string VideoKTH_3D::get_relative_video_key(const std::string &video_path) const
{
	// Convert absolute path to relative key matching JSON format
	// e.g., "/path/to/kth_organized/train/boxing/video.avi" -> "train/boxing/video.avi"
	std::string path = video_path;

	// Normalize path separators
	for (auto &c : path)
	{
		if (c == '\\')
			c = '/';
	}

	// Try to find train/ or test/ in the path
	size_t train_pos = path.find("train/");
	size_t test_pos = path.find("test/");

	if (train_pos != std::string::npos)
		return path.substr(train_pos);
	if (test_pos != std::string::npos)
		return path.substr(test_pos);

	// Fallback: use just the filename parts
	return path;
}

VideoKTH_3D::VideoKTH_3D(const std::string &video_folder_path, const std::string &hog_json_path,
						   const size_t &frame_per_video, const size_t &frame_gap, const size_t &threshold,
						   const size_t &sample_per_video, const size_t &grey_video,
						   std::string exp_name, const size_t &draw,
						   const size_t &frame_size_width, const size_t &frame_size_height, size_t max_read)
	: _video_folder_path(video_folder_path), _frame_per_video(frame_per_video),
	  _frame_gap(frame_gap), _frame_gap_counter(0), _grey_video(grey_video),
	  _sample_per_video(sample_per_video), _draw(draw),
	  _frame_size_width(frame_size_width), _frame_size_height(frame_size_height),
	  _exp_name(exp_name), _frame_preprocess(0), _frame_number(0), _threshold(threshold),
	  _cursor(0), _cursor_count(0), _label_count(0),
	  _shape({VIDEO_KTH3D_WIDTH, VIDEO_KTH3D_HEIGHT, VIDEO_KTH3D_DEPTH, VIDEO_KTH3D_CONV_DEPTH}),
	  _max_read(max_read), _hog_json_path(hog_json_path)
{
	_file_path = std::filesystem::current_path();

	std::string _exp_name_conf = _exp_name;
	if (_exp_name_conf.find('_') != std::string::npos)
		_exp_name_conf.erase(_exp_name_conf.rfind('_'));
	std::filesystem::create_directories(_file_path + "/Param_config/");

	// Load HOG JSON data (only once, shared across train/test instances)
	load_hog_json(hog_json_path);

	// Get the data from the data location
	for (const auto &file : std::filesystem::directory_iterator(_video_folder_path))
	{
		std::string _file_path = file.path();
		_action_list.push_back(_file_path.substr(_video_folder_path.length() + 1));

		for (const auto &_file : std::filesystem::directory_iterator(_file_path))
			_video_list.push_back(_file.path());
	}

	std::sort(_video_list.begin(), _video_list.end());
	std::sort(_action_list.begin(), _action_list.end());

	_size = _video_list.size();

	// Get the correct frame shape
	cv::VideoCapture capture(_video_list[0]);
	cv::Mat _size_frame;
	capture >> _size_frame;

	if (_grey_video == 1)
		cv::cvtColor(_size_frame, _size_frame, cv::COLOR_BGR2GRAY);

	size_t _width = _frame_size_width == 0 ? _size_frame.cols : _frame_size_width;
	size_t _height = _frame_size_height == 0 ? _size_frame.rows : _frame_size_height;
	size_t _depth = _size_frame.channels();
	size_t _conv_depth = _frame_per_video;
	_shape = Shape(std::vector<size_t>({_height, _width, _depth, _conv_depth}));
}

bool VideoKTH_3D::has_next() const
{
	return _cursor < size();
}

std::pair<std::string, Tensor<InputType>> VideoKTH_3D::next()
{
	_current_video_name = _video_list[_cursor];
	cv::VideoCapture capture(_current_video_name);

	if (!capture.isOpened())
		std::cout << "Unable to open file!" << std::endl;

	_frame_number = 0;
	int sz[3] = {_shape.dim(0), _shape.dim(1), _shape.dim(2)};

	size_t _label = assign_label_to_sample(_current_video_name);
	std::pair<std::string, Tensor<InputType>> out(std::to_string(static_cast<size_t>(_label)), _shape);

	// Get relative key for JSON lookup
	std::string rel_key = get_relative_video_key(_current_video_name);

	bool use_hog_frames = false;
	std::vector<int> target_frame_indices;

	if (_hog_data.find(rel_key) != _hog_data.end())
	{
		const VideoHOGData &vdata = _hog_data[rel_key];
		size_t group_idx = _cursor_count;

		if (group_idx < vdata.groups.size())
		{
			const VideoGroup &group = vdata.groups[group_idx];
			for (const auto &fb : group.frames)
			{
				target_frame_indices.push_back(fb.frame_idx);
			}
			use_hog_frames = true;
		}
	}

	if (use_hog_frames && !target_frame_indices.empty())
	{
		// Extract specific frames from the video based on HOG data
		int current_frame_pos = 0;
		size_t target_idx = 0;
		cv::Mat frame;

		// Sort target indices (should already be sorted, but just in case)
		std::sort(target_frame_indices.begin(), target_frame_indices.end());

		while (target_idx < target_frame_indices.size() && _frame_number < _frame_per_video)
		{
			int target = target_frame_indices[target_idx];

			// Seek to target frame
			if (current_frame_pos < target)
			{
				capture.set(cv::CAP_PROP_POS_FRAMES, target);
				current_frame_pos = target;
			}

			capture >> frame;
			if (frame.empty())
				break;

			current_frame_pos++;

			if (_frame_size_width != 0 || _frame_size_height != 0)
				cv::resize(frame, frame, cv::Size(_frame_size_width, _frame_size_height));

			if (_grey_video == 1)
				cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

			// Store frame in tensor
			for (int i = 0; i < frame.rows; i++)
				for (int j = 0; j < frame.cols; j++)
					for (int k = 0; k < frame.channels(); k++)
					{
						if (frame.channels() > 1)
							out.second.at(i, j, k, _frame_number) = (frame.at<cv::Vec3b>(i, j)[k]);
						else
							out.second.at(i, j, k, _frame_number) = (frame.at<unsigned char>(i, j));
					}

			_frame_number++;
			target_idx++;
		}

		// If we didn't get enough frames, pad with the last available frame
		while (_frame_number < _frame_per_video && !frame.empty())
		{
			for (int i = 0; i < frame.rows; i++)
				for (int j = 0; j < frame.cols; j++)
					for (int k = 0; k < frame.channels(); k++)
					{
						if (frame.channels() > 1)
							out.second.at(i, j, k, _frame_number) = (frame.at<cv::Vec3b>(i, j)[k]);
						else
							out.second.at(i, j, k, _frame_number) = (frame.at<unsigned char>(i, j));
					}
			_frame_number++;
		}
	}
	else
	{
		// Fallback: use standard Video sampling (random start + frame gap)
		cv::Mat frame(_shape.dim(2), sz, CV_32F, cv::Scalar(0));
		cv::Mat skipFrame(_shape.dim(2), sz, CV_32F, cv::Scalar(0));

		set_frame_gap(_cursor_count * 3, capture, skipFrame);

		capture >> frame;
		if (!frame.empty())
			if (_frame_size_width != 0 || _frame_size_height != 0)
				cv::resize(frame, frame, cv::Size(_frame_size_width, _frame_size_height));
		if (!frame.empty())
			if (_grey_video == 1)
				cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

		int fluct = 0;
		while (true)
		{
			cv::Mat next_frame(_shape.dim(2), sz, CV_32F, cv::Scalar(0));
			capture >> next_frame;

			if (frame.empty() || next_frame.empty() || _frame_number == _frame_per_video)
			{
				_frame_number = 0;
				break;
			}

			if (_frame_size_height != 0 || _frame_size_width != 0)
				cv::resize(next_frame, next_frame, cv::Size(_frame_size_width, _frame_size_height));

			if (_grey_video == 1)
				cv::cvtColor(next_frame, next_frame, cv::COLOR_BGR2GRAY);

			if (_threshold > 0)
				if (movement_threshold(frame, next_frame))
				{
					frame = next_frame;
					fluct = 0;
					continue;
				}

			if (_frame_gap > 0)
				set_frame_gap(_frame_gap, capture, skipFrame);

			if (!frame.empty())
			{
				for (int i = 0; i < frame.rows; i++)
					for (int j = 0; j < frame.cols; j++)
						for (int k = 0; k < frame.channels(); k++)
						{
							if (frame.channels() > 1)
								out.second.at(i, j, k, _frame_number) = (frame.at<cv::Vec3b>(i, j)[k]);
							else
								out.second.at(i, j, k, _frame_number) = (frame.at<unsigned char>(i, j));
						}
				_frame_number++;
			}
			fluct = 1;
			frame = next_frame;
		}
	}

	if (_sample_per_video > 0)
	{
		if (_cursor_count == _sample_per_video)
		{
			_cursor++;
			_cursor_count = 0;
		}
		_cursor_count++;
	}
	else
		_cursor++;

	if (_draw == 1)
		save_as_images(out);

	_frame_gap_counter = 0;
	capture.release();

	return out;
}

void VideoKTH_3D::save_as_images(std::pair<std::string, Tensor<InputType>> out)
{
	if (_current_video_name.find("train") != std::string::npos)
	{
		std::string name = _current_video_name.substr(_video_folder_path.length() + 1);
		std::string _action = name.substr(0, name.find("/"));
		std::filesystem::create_directories("Input_frames/" + _exp_name + "/train/" + _action + "/");
		Tensor<float>::draw_nonscaled_tensor(_file_path + "/Input_frames/" + _exp_name + "/train/" + _action + "/" + _action + "_" + std::to_string(_cursor) + "_" + std::to_string(_frame_number), out.second);
	}
	if (_current_video_name.find("test") != std::string::npos)
	{
		std::string name = _current_video_name.substr(_video_folder_path.length() + 1);
		std::string _action = name.substr(0, name.find("/"));
		std::filesystem::create_directories("Input_frames/" + _exp_name + "/test/" + _action + "/");
		Tensor<float>::draw_nonscaled_tensor(_file_path + "/Input_frames/" + _exp_name + "/test/" + _action + "/" + _action + "_" + std::to_string(_cursor) + "_" + std::to_string(_frame_number), out.second);
	}
}

uint32_t VideoKTH_3D::assign_label_to_sample(std::string _current_video_name)
{
	_current_video_name = _current_video_name.substr(_video_folder_path.length() + 1);
	std::string _action = _current_video_name.substr(0, _current_video_name.find("/"));
	_label_count = std::distance(_action_list.begin(), find(_action_list.begin(), _action_list.end(), _action));
	return _label_count;
}

void VideoKTH_3D::set_frame_gap(int _frame_gap, cv::VideoCapture capture, cv::Mat skipFrame)
{
	for (int i = 0; i < _frame_gap; i++)
		capture >> skipFrame;
}

bool VideoKTH_3D::movement_threshold(cv::Mat frame, cv::Mat next_frame)
{
	cv::Mat difference;
	difference = frame - next_frame;
	cv::Scalar sum = cv::sum(difference);
	if (sum(0) < _threshold)
		return true;
	return false;
}

cv::Mat VideoKTH_3D::frame_preprocess(int _frame_preprocess, cv::Mat frame, cv::Mat next_frame)
{
	if (_frame_preprocess == 0)
		return frame;
	else if (_frame_preprocess == 1)
	{
		cv::Mat difference;
		difference = frame - next_frame;
		cv::Scalar sum = cv::sum(difference);
		if (sum(0) > _threshold)
			return difference;
	}
	return frame;
}

void VideoKTH_3D::reset()
{
	_cursor = 0;
	_label_count = 0;
}

void VideoKTH_3D::close()
{
}

size_t VideoKTH_3D::size() const
{
	return std::min(_size, _max_read);
}

std::string VideoKTH_3D::to_string() const
{
	int _spv = _sample_per_video <= 0 ? 1 : _sample_per_video;
	if (_video_folder_path.find("train") != std::string::npos)
		set_sample_count(size() * _spv, 1);
	if (_video_folder_path.find("test") != std::string::npos)
		set_sample_count(size() * _spv, 2);

	return "VideoKTH_3D(" + _video_folder_path + ")[" + std::to_string(size()) + "]";
}

const Shape &VideoKTH_3D::shape() const
{
	return _shape;
}

uint32_t VideoKTH_3D::swap(uint32_t v)
{
	return 0;
}

const std::map<std::string, VideoKTH_3D::VideoHOGData>& VideoKTH_3D::get_hog_data()
{
	return _hog_data;
}