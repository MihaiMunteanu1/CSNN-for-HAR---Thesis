#ifndef _DATASET_IMAGE_SEQUENCE_KTH_H
#define _DATASET_IMAGE_SEQUENCE_KTH_H

#include <filesystem>
#include <string>
#include <vector>
#include <limits>
#include <map>

#include "Tensor.h"
#include "Input.h"

namespace dataset
{

/**
 * @brief Loads pre-cropped KTH person frames (.jpg) and groups them into
 *        temporal sequences for 3D convolution.
 *
 * Expected directory layout (produced by extract_frames_kth.py):
 *   kth_cropped/
 *     train/
 *       boxing/
 *         person01_boxing_d1_frame0042.jpg
 *         person01_boxing_d1_frame0047.jpg
 *         ...
 *     test/
 *       ...
 *
 * Frames from the same video (same person+action+scenario prefix) are
 * sorted by frame index and grouped into non-overlapping sequences of
 * `temporal_depth` consecutive frames.  Each sequence becomes one sample
 * with shape (height, width, 1, temporal_depth).
 *
 * The label returned by next() is the action name string (e.g. "boxing").
 */
class ImageSequenceKTH : public Input
{

public:
    ImageSequenceKTH(const std::string &folder_path,
                     size_t temporal_depth = 5,
                     size_t frame_size_width = 48,
                     size_t frame_size_height = 80,
                     size_t grey = 1,
                     size_t max_samples_per_video = 0,
                     size_t max_read = std::numeric_limits<size_t>::max());

    virtual bool has_next() const override;
    virtual std::pair<std::string, Tensor<InputType>> next() override;
    virtual void reset() override;
    virtual void close() override;
    virtual std::string to_string() const override;
    virtual const Shape &shape() const override;

private:
    struct Sample {
        std::string action;                  // label: "boxing", "running", etc.
        std::vector<std::string> frame_paths; // temporal_depth file paths
    };

    void build_samples();

    std::string _folder_path;
    size_t _temporal_depth;
    size_t _frame_size_width;
    size_t _frame_size_height;
    size_t _grey;
    size_t _max_samples_per_video;
    size_t _max_read;

    std::vector<Sample> _samples;
    size_t _cursor;

    Shape _shape;
    bool _is_train;
};

} // namespace dataset

#endif
