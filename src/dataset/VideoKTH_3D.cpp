#include "dataset/VideoKTH_3D.h"
#include <iostream>
#include <sstream>
#include <cstring>

using namespace dataset;

std::map<std::string, VideoKTH_3D::VideoHOGData> VideoKTH_3D::_hog_data;
bool VideoKTH_3D::_hog_data_loaded = false;
std::map<size_t, std::pair<std::string, size_t>> VideoKTH_3D::_train_sample_mapping;
std::map<size_t, std::pair<std::string, size_t>> VideoKTH_3D::_val_sample_mapping;
std::map<size_t, std::pair<std::string, size_t>> VideoKTH_3D::_test_sample_mapping;
size_t VideoKTH_3D::_train_sample_counter = 0;
size_t VideoKTH_3D::_val_sample_counter = 0;
size_t VideoKTH_3D::_test_sample_counter = 0;

// NPY cache static members
bool VideoKTH_3D::_npy_loaded = false;
std::vector<uint8_t> VideoKTH_3D::_npy_frames;
size_t VideoKTH_3D::_npy_N = 0;
size_t VideoKTH_3D::_npy_T = 0;
size_t VideoKTH_3D::_npy_H = 0;
size_t VideoKTH_3D::_npy_W = 0;
std::vector<int> VideoKTH_3D::_npy_labels;
std::vector<std::string> VideoKTH_3D::_npy_actions;
std::vector<std::string> VideoKTH_3D::_npy_video_keys;
std::vector<size_t> VideoKTH_3D::_npy_group_idx;
std::vector<std::string> VideoKTH_3D::_npy_splits;
bool VideoKTH_3D::_npy_has_bboxes = false;
std::vector<int> VideoKTH_3D::_npy_bboxes;
std::vector<size_t> VideoKTH_3D::_train_global_indices;
std::vector<size_t> VideoKTH_3D::_val_global_indices;
std::vector<size_t> VideoKTH_3D::_test_global_indices;


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

static bool load_npy_uint8_4d(const std::string &path,
							  std::vector<uint8_t> &out_data,
							  size_t &N, size_t &T, size_t &H, size_t &W)
{
	std::ifstream f(path, std::ios::binary);
	if (!f.is_open())
	{
		std::cerr << "NPY: cannot open " << path << std::endl;
		return false;
	}

	char magic[6];
	f.read(magic, 6);
	if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
	{
		std::cerr << "NPY: bad magic in " << path << std::endl;
		return false;
	}

	uint8_t ver_major = 0, ver_minor = 0;
	f.read(reinterpret_cast<char *>(&ver_major), 1);
	f.read(reinterpret_cast<char *>(&ver_minor), 1);
	(void)ver_minor;

	size_t header_len = 0;
	if (ver_major == 1)
	{
		uint16_t hl = 0;
		f.read(reinterpret_cast<char *>(&hl), 2);
		header_len = hl;
	}
	else
	{
		uint32_t hl = 0;
		f.read(reinterpret_cast<char *>(&hl), 4);
		header_len = hl;
	}

	std::string header(header_len, '\0');
	f.read(&header[0], header_len);

	// Parse 'shape': (N, T, H, W).
	size_t shape_pos = header.find("'shape'");
	if (shape_pos == std::string::npos)
		shape_pos = header.find("\"shape\"");
	if (shape_pos == std::string::npos)
	{
		std::cerr << "NPY: 'shape' not found in header" << std::endl;
		return false;
	}
	size_t open_paren = header.find('(', shape_pos);
	size_t close_paren = header.find(')', open_paren);
	if (open_paren == std::string::npos || close_paren == std::string::npos)
		return false;
	std::string shape_str = header.substr(open_paren + 1, close_paren - open_paren - 1);

	std::vector<size_t> dims;
	size_t pos = 0;
	while (pos < shape_str.size())
	{
		while (pos < shape_str.size() && (shape_str[pos] == ' ' || shape_str[pos] == ','))
			pos++;
		if (pos >= shape_str.size())
			break;
		size_t end = pos;
		while (end < shape_str.size() && std::isdigit(static_cast<unsigned char>(shape_str[end])))
			end++;
		if (end > pos)
			dims.push_back(static_cast<size_t>(std::stoul(shape_str.substr(pos, end - pos))));
		pos = end;
	}

	if (dims.size() != 4)
	{
		std::cerr << "NPY: expected 4D, got " << dims.size() << "D" << std::endl;
		return false;
	}
	N = dims[0]; T = dims[1]; H = dims[2]; W = dims[3];

	if (header.find("'|u1'") == std::string::npos &&
		header.find("\"|u1\"") == std::string::npos)
	{
		std::cerr << "NPY: expected dtype |u1 (uint8) in " << path << std::endl;
		return false;
	}

	size_t total = N * T * H * W;
	out_data.resize(total);
	f.read(reinterpret_cast<char *>(out_data.data()), total);
	if (!f)
	{
		std::cerr << "NPY: failed to read " << total << " bytes" << std::endl;
		return false;
	}
	return true;
}


static std::string json_extract_string(const std::string &obj, const std::string &key)
{
	std::string search = "\"" + key + "\"";
	size_t k = obj.find(search);
	if (k == std::string::npos) return "";
	size_t colon = obj.find(':', k);
	if (colon == std::string::npos) return "";
	size_t q1 = obj.find('"', colon);
	if (q1 == std::string::npos) return "";
	size_t q2 = obj.find('"', q1 + 1);
	if (q2 == std::string::npos) return "";
	return obj.substr(q1 + 1, q2 - q1 - 1);
}

static int json_extract_int(const std::string &obj, const std::string &key)
{
	std::string search = "\"" + key + "\"";
	size_t k = obj.find(search);
	if (k == std::string::npos) return 0;
	size_t colon = obj.find(':', k);
	if (colon == std::string::npos) return 0;
	size_t p = colon + 1;
	while (p < obj.size() && (obj[p] == ' ' || obj[p] == '\t')) p++;
	std::string num;
	while (p < obj.size() &&
		   (std::isdigit(static_cast<unsigned char>(obj[p])) || obj[p] == '-'))
	{
		num += obj[p];
		p++;
	}
	return num.empty() ? 0 : std::stoi(num);
}

// Parse a "bboxes": [[x,y,w,h], [x,y,w,h], ...] field from a single sample
// object string. Returns a flat vector of length 4*T (or empty if absent /
// malformed). The script that produces the metadata always emits T inner
// arrays of exactly 4 ints, so we don't try to recover from partial parses.
static std::vector<int> json_extract_bboxes(const std::string &obj)
{
	std::vector<int> out;
	size_t k = obj.find("\"bboxes\"");
	if (k == std::string::npos) return out;
	size_t colon = obj.find(':', k);
	if (colon == std::string::npos) return out;
	size_t outer = obj.find('[', colon);
	if (outer == std::string::npos) return out;

	// Walk the outer array, picking each inner [x,y,w,h] tuple.
	size_t p = outer + 1;
	while (p < obj.size())
	{
		while (p < obj.size() &&
			   (obj[p] == ' ' || obj[p] == '\t' || obj[p] == '\n' ||
				obj[p] == '\r' || obj[p] == ','))
			p++;
		if (p >= obj.size() || obj[p] == ']') break;
		if (obj[p] != '[') { p++; continue; }
		p++;
		int parsed = 0;
		while (p < obj.size() && parsed < 4)
		{
			while (p < obj.size() &&
				   (obj[p] == ' ' || obj[p] == '\t' || obj[p] == ',' ||
					obj[p] == '\n' || obj[p] == '\r'))
				p++;
			if (p >= obj.size() || obj[p] == ']') break;
			std::string num;
			while (p < obj.size() &&
				   (std::isdigit(static_cast<unsigned char>(obj[p])) || obj[p] == '-'))
			{
				num += obj[p];
				p++;
			}
			if (!num.empty())
			{
				out.push_back(std::stoi(num));
				parsed++;
			}
			else
			{
				p++;
			}
		}
		// Skip to the inner ']'.
		while (p < obj.size() && obj[p] != ']') p++;
		if (p < obj.size()) p++;
	}
	return out;
}

static std::vector<std::string> json_split_top_level_objects(const std::string &arr_str)
{
	std::vector<std::string> out;
	int depth = 0;
	size_t start = std::string::npos;
	for (size_t i = 0; i < arr_str.size(); i++)
	{
		char c = arr_str[i];
		if (c == '{')
		{
			if (depth == 0) start = i;
			depth++;
		}
		else if (c == '}')
		{
			depth--;
			if (depth == 0 && start != std::string::npos)
			{
				out.push_back(arr_str.substr(start, i - start + 1));
				start = std::string::npos;
			}
		}
	}
	return out;
}

void VideoKTH_3D::load_npy_cache(const std::string &npy_path)
{
	if (_npy_loaded)
		return;

	if (!load_npy_uint8_4d(npy_path, _npy_frames,
						   _npy_N, _npy_T, _npy_H, _npy_W))
	{
		std::cerr << "VideoKTH_3D: NPY load FAILED for " << npy_path << std::endl;
		_npy_loaded = true;
		return;
	}

	std::cout << "VideoKTH_3D: NPY loaded " << npy_path
			  << "  shape=(" << _npy_N << "," << _npy_T << ","
			  << _npy_H << "," << _npy_W << ")  "
			  << (_npy_frames.size() / 1e6) << " MB" << std::endl;

	std::string meta_path = npy_path;
	if (meta_path.size() >= 4 && meta_path.substr(meta_path.size() - 4) == ".npy")
		meta_path = meta_path.substr(0, meta_path.size() - 4) + ".json";

	std::ifstream mf(meta_path);
	if (!mf.is_open())
	{
		std::cerr << "VideoKTH_3D: NPY metadata not found: " << meta_path << std::endl;
		_npy_loaded = true;
		return;
	}

	std::stringstream buf;
	buf << mf.rdbuf();
	std::string content = buf.str();
	mf.close();

	size_t samples_pos = content.find("\"samples\"");
	if (samples_pos == std::string::npos)
	{
		std::cerr << "VideoKTH_3D: 'samples' key not found in " << meta_path << std::endl;
		_npy_loaded = true;
		return;
	}
	size_t arr_start = content.find('[', samples_pos);
	if (arr_start == std::string::npos)
	{
		_npy_loaded = true;
		return;
	}

	int depth = 1;
	size_t arr_end = arr_start + 1;
	while (arr_end < content.size() && depth > 0)
	{
		if (content[arr_end] == '[') depth++;
		else if (content[arr_end] == ']') depth--;
		arr_end++;
	}
	std::string arr_str = content.substr(arr_start + 1, arr_end - arr_start - 2);

	std::vector<std::string> sample_objs = json_split_top_level_objects(arr_str);
	if (sample_objs.size() != _npy_N)
	{
		std::cerr << "VideoKTH_3D: WARNING metadata samples (" << sample_objs.size()
				  << ") != NPY N (" << _npy_N << ")" << std::endl;
	}

	_npy_labels.reserve(sample_objs.size());
	_npy_video_keys.reserve(sample_objs.size());
	_npy_actions.reserve(sample_objs.size());
	_npy_group_idx.reserve(sample_objs.size());
	_npy_splits.reserve(sample_objs.size());
	_npy_bboxes.reserve(sample_objs.size() * _npy_T * 4);

	const size_t expected_bbox_len = _npy_T * 4;
	bool any_missing_bbox = false;

	for (const auto &obj : sample_objs)
	{
		_npy_labels.push_back(json_extract_int(obj, "label_idx"));
		_npy_video_keys.push_back(json_extract_string(obj, "video_key"));
		_npy_actions.push_back(json_extract_string(obj, "action"));
		_npy_group_idx.push_back(static_cast<size_t>(json_extract_int(obj, "group_idx")));
		_npy_splits.push_back(json_extract_string(obj, "split"));

		std::vector<int> bb = json_extract_bboxes(obj);
		if (bb.size() == expected_bbox_len)
		{
			_npy_bboxes.insert(_npy_bboxes.end(), bb.begin(), bb.end());
		}
		else
		{
			_npy_bboxes.insert(_npy_bboxes.end(), expected_bbox_len, 0);
			if (!bb.empty()) any_missing_bbox = true;
		}
	}

	size_t expected_total = sample_objs.size() * expected_bbox_len;
	_npy_has_bboxes = (_npy_bboxes.size() == expected_total) &&
					  (expected_total > 0) && !any_missing_bbox &&
					  (json_extract_bboxes(sample_objs[0]).size() == expected_bbox_len);

	// Per-split global-index tables for HOGSampler3D lookups.
	_train_global_indices.clear();
	_val_global_indices.clear();
	_test_global_indices.clear();
	_train_global_indices.reserve(_npy_splits.size());
	_val_global_indices.reserve(_npy_splits.size());
	_test_global_indices.reserve(_npy_splits.size());
	for (size_t i = 0; i < _npy_splits.size(); i++)
	{
		const std::string &s = _npy_splits[i];
		if (s == "train")     _train_global_indices.push_back(i);
		else if (s == "val")  _val_global_indices.push_back(i);
		else if (s == "test") _test_global_indices.push_back(i);
	}

	std::cout << "VideoKTH_3D: NPY metadata loaded "
			  << _npy_labels.size() << " sample entries"
			  << (_npy_has_bboxes ? " (with bboxes)" : " (no bboxes)") << std::endl;

	_npy_loaded = true;
}

// ---------------------------------------------------------------------------

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

	// Parse the JSON manually
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
		// Find next video key (ex "train/boxing/video.avi")
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
	//ex "/path/to/kth_organized/train/boxing/video.avi" -> "train/boxing/video.avi"
	std::string path = video_path;

	for (auto &c : path)
	{
		if (c == '\\')
			c = '/';
	}

	size_t train_pos = path.find("train/");
	size_t test_pos = path.find("test/");

	if (train_pos != std::string::npos)
		return path.substr(train_pos);
	if (test_pos != std::string::npos)
		return path.substr(test_pos);

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
	  _max_read(max_read), _hog_json_path(hog_json_path), _use_npy_cache(false)
{


	_file_path = std::filesystem::current_path();

	std::string _exp_name_conf = _exp_name;
	if (_exp_name_conf.find('_') != std::string::npos)
		_exp_name_conf.erase(_exp_name_conf.rfind('_'));
	std::filesystem::create_directories(_file_path + "/Param_config/");

	// Detect split from the folder path
	{
		std::string p = _video_folder_path;
		for (auto &c : p) if (c == '\\') c = '/';
		if      (p.find("/train") != std::string::npos) _split_name = "train";
		else if (p.find("/val")   != std::string::npos) _split_name = "val";
		else if (p.find("/test")  != std::string::npos) _split_name = "test";
		else                                            _split_name = "train";
	}

	// Auto-detect: .npy → load pre-extracted frame cache, .json → legacy bbox path.
	bool path_is_npy = (hog_json_path.size() >= 4) &&
					   (hog_json_path.substr(hog_json_path.size() - 4) == ".npy");

	if (path_is_npy)
	{
		_use_npy_cache = true;
		load_npy_cache(hog_json_path);

		_split_sample_indices.clear();
		_split_sample_indices.reserve(_npy_labels.size());
		for (size_t i = 0; i < _npy_labels.size(); i++)
		{
			if (_npy_splits[i] == _split_name)
				_split_sample_indices.push_back(i);
		}

		_action_list = {"boxing", "handclapping", "handwaving",
						"jogging", "running", "walking"};

		_size = static_cast<uint32_t>(_split_sample_indices.size());

		size_t _width = _frame_size_width == 0 ? _npy_W : _frame_size_width;
		size_t _height = _frame_size_height == 0 ? _npy_H : _frame_size_height;
		size_t _depth = 1;
		size_t _conv_depth = _frame_per_video;
		_shape = Shape(std::vector<size_t>({_height, _width, _depth, _conv_depth}));

		std::cout << "VideoKTH_3D[" << _split_name << "]: "
				  << _split_sample_indices.size() << " samples from NPY cache." << std::endl;
		return;
	}

	load_hog_json(hog_json_path);

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

std::pair<std::string, Tensor<InputType>> VideoKTH_3D::next_from_cache()
{
	size_t local_idx = static_cast<size_t>(_cursor);
	size_t global_idx = _split_sample_indices[local_idx];

	const std::string &action = _npy_actions[global_idx];
	const std::string &video_key = _npy_video_keys[global_idx];
	size_t group_idx = _npy_group_idx[global_idx];

	{
		std::string suffix = video_key;
		size_t slash = video_key.find('/');
		if (slash != std::string::npos)
			suffix = video_key.substr(slash + 1);
		_current_video_name = _video_folder_path + suffix;
	}

	std::pair<std::string, Tensor<InputType>> out(action, _shape);

	const size_t T = _npy_T;
	const size_t H = _npy_H;
	const size_t W = _npy_W;
	const size_t per_sample = T * H * W;
	const uint8_t *base = _npy_frames.data() + global_idx * per_sample;

	for (size_t t = 0; t < T && t < static_cast<size_t>(_frame_per_video); t++)
	{
		const uint8_t *fr = base + t * H * W;
		for (size_t i = 0; i < H; i++)
			for (size_t j = 0; j < W; j++)
				out.second.at(i, j, 0, t) = static_cast<InputType>(fr[i * W + j]);
	}

	if (_split_name == "train")
	{
		_train_sample_mapping[_train_sample_counter] = {video_key, group_idx};
		_train_sample_counter++;
	}
	else if (_split_name == "val")
	{
		_val_sample_mapping[_val_sample_counter] = {video_key, group_idx};
		_val_sample_counter++;
	}
	else
	{
		_test_sample_mapping[_test_sample_counter] = {video_key, group_idx};
		_test_sample_counter++;
	}

	_cursor++;
	if (_draw == 1)
		save_as_images(out);
	return out;
}

std::pair<std::string, Tensor<InputType>> VideoKTH_3D::next()
{
	if (_use_npy_cache)
		return next_from_cache();

	_current_video_name = _video_list[_cursor];
	cv::VideoCapture capture(_current_video_name);

	if (!capture.isOpened())
		std::cout << "Unable to open file!" << std::endl;

	_frame_number = 0;
	int sz[3] = {static_cast<int>(_shape.dim(0)), static_cast<int>(_shape.dim(1)), static_cast<int>(_shape.dim(2))};

	assign_label_to_sample(_current_video_name);


	std::string _action_name;
	{
		std::string rel = _current_video_name.substr(_video_folder_path.length() + 1);
		size_t slash = rel.find("/");
		_action_name = (slash != std::string::npos) ? rel.substr(0, slash) : rel;
	}
	std::pair<std::string, Tensor<InputType>> out(_action_name, _shape);

	std::string rel_key = get_relative_video_key(_current_video_name);

	bool use_hog_frames = false;
	std::vector<int> target_frame_indices;

	std::vector<std::tuple<int, int, int, int>> target_bboxes;


	size_t group_idx = _cursor_count;
	if (_hog_data.find(rel_key) != _hog_data.end())
	{
		const VideoHOGData &vdata = _hog_data[rel_key];
		if (!vdata.groups.empty())
		{
			group_idx = _cursor_count % vdata.groups.size();
			const VideoGroup &group = vdata.groups[group_idx];
			for (const auto &fb : group.frames)
			{
				target_frame_indices.push_back(fb.frame_idx);
				if (!fb.bboxes.empty())
					target_bboxes.push_back(fb.bboxes[0]);
				else
					target_bboxes.push_back(std::make_tuple(0, 0, 0, 0));
			}
			use_hog_frames = true;
		}
	}

	{
		if (_split_name == "train")
		{
			_train_sample_mapping[_train_sample_counter] = {rel_key, group_idx};
			_train_sample_counter++;
		}
		else if (_split_name == "val")
		{
			_val_sample_mapping[_val_sample_counter] = {rel_key, group_idx};
			_val_sample_counter++;
		}
		else
		{
			_test_sample_mapping[_test_sample_counter] = {rel_key, group_idx};
			_test_sample_counter++;
		}
	}

	if (use_hog_frames && !target_frame_indices.empty())
	{
		int current_frame_pos = 0;
		size_t target_idx = 0;
		cv::Mat frame;


		{
			std::vector<std::pair<int, std::tuple<int, int, int, int>>> paired;
			paired.reserve(target_frame_indices.size());
			for (size_t i = 0; i < target_frame_indices.size(); i++)
				paired.emplace_back(target_frame_indices[i], target_bboxes[i]);
			std::sort(paired.begin(), paired.end(),
				[](const auto &a, const auto &b) { return a.first < b.first; });
			for (size_t i = 0; i < paired.size(); i++)
			{
				target_frame_indices[i] = paired[i].first;
				target_bboxes[i] = paired[i].second;
			}
		}

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


			if (frame.cols != 160 || frame.rows != 120)
				cv::resize(frame, frame, cv::Size(160, 120));

			auto [bx, by, bw, bh] = target_bboxes[target_idx];
			if (bw > 0 && bh > 0)
			{
				float pad = 1.25f; // padd 1.25%
				float cx = bx + bw / 2.0f;
				float cy = by + bh / 2.0f;
				int new_w = std::max(1, (int)std::ceil(bw * pad));
				int new_h = std::max(1, (int)std::ceil(bh * pad));
				int new_x = std::max(0, (int)(cx - new_w / 2.0f));
				int new_y = std::max(0, (int)(cy - new_h / 2.0f));

				new_x = std::max(0, std::min(new_x, frame.cols - 1));
				new_y = std::max(0, std::min(new_y, frame.rows - 1));
				new_w = std::max(1, std::min(new_w, frame.cols - new_x));
				new_h = std::max(1, std::min(new_h, frame.rows - new_y));

				if (new_w > 0 && new_h > 0) {
					cv::Rect roi(new_x, new_y, new_w, new_h);
					frame = frame(roi);
				}
			}

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
        _cursor_count++;
        if (_cursor_count >= _sample_per_video)
        {
            _cursor++;
            _cursor_count = 0;
        }
    }
    else
    {
        _cursor++;
    }

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

void VideoKTH_3D::set_frame_gap(int _frame_gap, cv::VideoCapture& capture, cv::Mat& skipFrame)
{
	for (int i = 0; i < _frame_gap; i++)
		capture >> skipFrame;
}

bool VideoKTH_3D::movement_threshold(cv::Mat frame, cv::Mat next_frame)
{
    if (frame.empty() || next_frame.empty()) return true;
    cv::Mat difference;
    cv::absdiff(frame, next_frame, difference);
    cv::Scalar s = cv::sum(difference);
    double motion = 0.0;
    for (int c = 0; c < difference.channels(); c++) motion += s[c];
    return motion < static_cast<double>(_threshold);
//	cv::Mat difference;
//	difference = frame - next_frame;
//	cv::Scalar sum = cv::sum(difference);
//	if (sum(0) < _threshold)
//		return true;
//	return false;
}

void VideoKTH_3D::reset_sample_mappings()
{
    _train_sample_mapping.clear();
    _val_sample_mapping.clear();
    _test_sample_mapping.clear();
    _train_sample_counter = 0;
    _val_sample_counter = 0;
    _test_sample_counter = 0;
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
	_cursor_count = 0;
	_label_count = 0;


	if (_split_name == "train")
	{
		_train_sample_mapping.clear();
		_train_sample_counter = 0;
	}
	else if (_split_name == "val")
	{
		_val_sample_mapping.clear();
		_val_sample_counter = 0;
	}
	else
	{
		_test_sample_mapping.clear();
		_test_sample_counter = 0;
	}
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
	if (_use_npy_cache) _spv = 1;
	if (_split_name == "train")
		set_sample_count(size() * _spv, 1);
	else if (_split_name == "test")
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

const std::map<size_t, std::pair<std::string, size_t>>& VideoKTH_3D::get_train_sample_mapping()
{
	return _train_sample_mapping;
}

const std::map<size_t, std::pair<std::string, size_t>>& VideoKTH_3D::get_test_sample_mapping()
{
	return _test_sample_mapping;
}

const std::map<size_t, std::pair<std::string, size_t>>& VideoKTH_3D::get_val_sample_mapping()
{
	return _val_sample_mapping;
}

bool VideoKTH_3D::has_bboxes()
{
	return _npy_has_bboxes;
}

static std::tuple<int, int, int, int> bbox_at(
	const std::vector<size_t> &split_indices,
	const std::vector<int> &flat_bboxes,
	size_t T, size_t local_idx, size_t t)
{
	if (!VideoKTH_3D::has_bboxes()) return {0, 0, 0, 0};
	if (local_idx >= split_indices.size()) return {0, 0, 0, 0};
	if (t >= T) return {0, 0, 0, 0};
	size_t global_idx = split_indices[local_idx];
	size_t base = (global_idx * T + t) * 4;
	if (base + 3 >= flat_bboxes.size()) return {0, 0, 0, 0};
	return {flat_bboxes[base + 0], flat_bboxes[base + 1],
			flat_bboxes[base + 2], flat_bboxes[base + 3]};
}

std::tuple<int, int, int, int> VideoKTH_3D::get_train_sample_bbox(size_t local_idx, size_t t)
{
	return bbox_at(_train_global_indices, _npy_bboxes, _npy_T, local_idx, t);
}

std::tuple<int, int, int, int> VideoKTH_3D::get_val_sample_bbox(size_t local_idx, size_t t)
{
	return bbox_at(_val_global_indices, _npy_bboxes, _npy_T, local_idx, t);
}

std::tuple<int, int, int, int> VideoKTH_3D::get_test_sample_bbox(size_t local_idx, size_t t)
{
	return bbox_at(_test_global_indices, _npy_bboxes, _npy_T, local_idx, t);
}