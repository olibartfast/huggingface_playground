#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/**
 * @brief Reads frames from a video file, sampling at 1 FPS.
 *
 * @param video_path Path to the video file.
 * @param target_frames Maximum number of frames/seconds to read.
 * @return std::vector<cv::Mat> Vector of read frames (RGB).
 */
std::vector<cv::Mat> read_video_frames(const std::string &video_path,
                                       int target_frames);

/**
 * @brief Pads a sequence of frames to a target length by duplicating the last
 * frame.
 *
 * @param frames Input frames.
 * @param target_length Desired length of the frame sequence.
 * @return std::vector<cv::Mat> Padded frame sequence.
 */
std::vector<cv::Mat> pad_video_frames(const std::vector<cv::Mat> &frames,
                                      int target_length);
