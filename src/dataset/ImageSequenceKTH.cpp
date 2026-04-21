#include "dataset/ImageSequenceKTH.h"
#include "tool/Operations.h"
#include <iostream>
#include <algorithm>
#include <map>
#include <opencv2/opencv.hpp>

using namespace dataset;

// ---------------------------------------------------------------------------
// Helper: extract video key from filename
//   "person01_boxing_d1_frame0042.jpg" -> "person01_boxing_d1"
// ---------------------------------------------------------------------------
static std::string video_key_from_filename(const std::string &filename)
{
    // Strip extension
    std::string base = filename;
    size_t dot = base.rfind('.');
    if (dot != std::string::npos)
        base = base.substr(0, dot);

    // Strip _frameXXXX suffix
    size_t frame_pos = base.rfind("_frame");
    if (frame_pos != std::string::npos)
        base = base.substr(0, frame_pos);

    return base;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
ImageSequenceKTH::ImageSequenceKTH(const std::string &folder_path,
                                   size_t temporal_depth,
                                   size_t frame_size_width,
                                   size_t frame_size_height,
                                   size_t grey,
                                   size_t max_samples_per_video,
                                   size_t max_read)
    : _folder_path(folder_path),
      _temporal_depth(temporal_depth),
      _frame_size_width(frame_size_width),
      _frame_size_height(frame_size_height),
      _grey(grey),
      _max_samples_per_video(max_samples_per_video),
      _max_read(max_read),
      _cursor(0),
      _shape({1, 1, 1, 1})
{
    _is_train = (_folder_path.find("train") != std::string::npos);

    // Determine shape from first image or from params
    size_t depth = _grey == 1 ? 1 : 3;
    _shape = Shape(std::vector<size_t>({_frame_size_height, _frame_size_width, depth, _temporal_depth}));

    build_samples();

    std::cout << "ImageSequenceKTH: " << (_is_train ? "train" : "test")
              << " — " << _samples.size() << " samples"
              << " (temporal_depth=" << _temporal_depth
              << ", " << _frame_size_width << "x" << _frame_size_height << ")"
              << std::endl;
}

// ---------------------------------------------------------------------------
// Build the sample list: scan folders, group by video, chunk into sequences
// ---------------------------------------------------------------------------
void ImageSequenceKTH::build_samples()
{
    _samples.clear();

    if (!std::filesystem::is_directory(_folder_path))
    {
        std::cerr << "WARNING: ImageSequenceKTH — directory not found: " << _folder_path << std::endl;
        return;
    }

    // Iterate over action subdirectories
    for (const auto &action_entry : std::filesystem::directory_iterator(_folder_path))
    {
        if (!action_entry.is_directory())
            continue;

        std::string action = action_entry.path().filename().string();

        // Collect all jpg files and group by video key
        // key = "person01_boxing_d1", value = sorted list of full paths
        std::map<std::string, std::vector<std::string>> video_frames;

        for (const auto &file_entry : std::filesystem::directory_iterator(action_entry.path()))
        {
            if (!file_entry.is_regular_file())
                continue;

            std::string fname = file_entry.path().filename().string();
            std::string ext = file_entry.path().extension().string();

            // Accept common image formats
            if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".bmp")
                continue;

            std::string vkey = video_key_from_filename(fname);
            video_frames[vkey].push_back(file_entry.path().string());
        }

        // For each video: sort frames by name (lexicographic = by frame index
        // since frame numbers are zero-padded), then chunk into groups of
        // temporal_depth
        for (auto &[vkey, paths] : video_frames)
        {
            std::sort(paths.begin(), paths.end());

            size_t num_full_groups = paths.size() / _temporal_depth;
            if (num_full_groups == 0)
                continue;

            size_t groups_to_use = num_full_groups;
            if (_max_samples_per_video > 0 && groups_to_use > _max_samples_per_video)
                groups_to_use = _max_samples_per_video;

            for (size_t g = 0; g < groups_to_use; ++g)
            {
                Sample s;
                s.action = action;
                for (size_t f = 0; f < _temporal_depth; ++f)
                    s.frame_paths.push_back(paths[g * _temporal_depth + f]);
                _samples.push_back(std::move(s));
            }
        }
    }

    // Sort samples by action then by path for deterministic ordering
    std::sort(_samples.begin(), _samples.end(), [](const Sample &a, const Sample &b) {
        if (a.action != b.action) return a.action < b.action;
        return a.frame_paths[0] < b.frame_paths[0];
    });
}

// ---------------------------------------------------------------------------
// Input interface
// ---------------------------------------------------------------------------

bool ImageSequenceKTH::has_next() const
{
    size_t effective_size = std::min(_samples.size(), _max_read);
    return _cursor < effective_size;
}

std::pair<std::string, Tensor<InputType>> ImageSequenceKTH::next()
{
    const Sample &sample = _samples[_cursor];
    std::pair<std::string, Tensor<InputType>> out(sample.action, _shape);

    for (size_t t = 0; t < sample.frame_paths.size(); ++t)
    {
        cv::Mat frame = cv::imread(sample.frame_paths[t], cv::IMREAD_COLOR);
        if (frame.empty())
        {
            std::cerr << "WARNING: Cannot read " << sample.frame_paths[t] << std::endl;
            continue;
        }

        if (_frame_size_width != 0 && _frame_size_height != 0)
            cv::resize(frame, frame, cv::Size(_frame_size_width, _frame_size_height));

        if (_grey == 1)
            cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

        for (int i = 0; i < frame.rows; i++)
            for (int j = 0; j < frame.cols; j++)
                for (int k = 0; k < frame.channels(); k++)
                {
                    if (frame.channels() > 1)
                        out.second.at(i, j, k, t) = (frame.at<cv::Vec3b>(i, j)[k]);
                    else
                        out.second.at(i, j, k, t) = (frame.at<unsigned char>(i, j));
                }
    }

    _cursor++;
    return out;
}

void ImageSequenceKTH::reset()
{
    _cursor = 0;
}

void ImageSequenceKTH::close()
{
}

std::string ImageSequenceKTH::to_string() const
{
    size_t effective_size = std::min(_samples.size(), _max_read);

    if (_is_train)
        set_sample_count(static_cast<int>(effective_size), 1);
    else
        set_sample_count(static_cast<int>(effective_size), 2);

    return "ImageSequenceKTH(" + _folder_path + ")[" + std::to_string(effective_size) + "]";
}

const Shape &ImageSequenceKTH::shape() const
{
    return _shape;
}
