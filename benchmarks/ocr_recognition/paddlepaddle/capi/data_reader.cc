// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "data_reader.h"
#include <fstream>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <sys/stat.h>

namespace paddle {

namespace {

cv::Mat resize(cv::Mat img, int target_width, int target_height) {
  cv::Mat resized_img;
  cv::resize(img, resized_img, cv::Size(target_width, target_height), 0, 0,
             cv::INTER_LANCZOS4);
  return resized_img;
}

static void split(const std::string &str, char sep,
                  std::vector<std::string> *pieces) {
  pieces->clear();
  if (str.empty()) {
    return;
  }
  size_t pos = 0;
  size_t next = str.find(sep, pos);
  while (next != std::string::npos) {
    pieces->push_back(str.substr(pos, next - pos));
    pos = next + 1;
    next = str.find(sep, pos);
  }
  if (!str.substr(pos).empty()) {
    pieces->push_back(str.substr(pos));
  }
}

} // namespace

DataReader::DataReader(const std::string &data_list_path,
                       const std::string &data_dir_path, int image_width,
                       int image_height, int channels)
    : data_list_path(data_list_path), data_dir_path(data_dir_path),
      file(data_list_path), image_width(image_width),
      image_height(image_height), channels(channels) {
  if (!file.is_open()) {
    throw std::invalid_argument("Cannot open data list file " + data_list_path);
  }

  if (data_dir_path.empty()) {
    throw std::invalid_argument("Data directory must be set to use imagenet.");
  }

  struct stat sb;
  if (stat(data_dir_path.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode)) {
    throw std::invalid_argument("Data directory does not exist.");
  }

  if (channels != 1) {
    throw std::invalid_argument("Only 1 channel image loading supported");
  }
}

bool DataReader::NextBatch(float *input,
                           std::vector<std::vector<int64_t>> &gt_label,
                           int batch_size) {
  gt_label.clear();

  std::string line;
  for (int i = 0; i < batch_size; i++) {
    if (!std::getline(file, line))
      return false;

    std::vector<std::string> pieces;
    split(line, sep, &pieces);
    if (pieces.size() != 4) {
      std::stringstream ss;
      ss << "invalid number of separators '" << sep << "' found in line " << i
         << ":'" << line << "' of file " << data_list_path;
      throw std::runtime_error(ss.str());
    }

    auto width = pieces.at(0);
    auto height = pieces.at(1);
    auto filename = data_dir_path + "/" + pieces.at(2);
    auto indices_str = pieces.at(3);

    // read image
    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    if (image.data == nullptr) {
      std::string error_msg = "Couldn't open file " + filename;
      throw std::runtime_error(error_msg);
    }

    cv::Mat image_resized = image;
    if (std::stoi(width) != image_width || std::stoi(height) != image_height) {
      image_resized = resize(image, image_width, image_height);
    }

    cv::Mat fimage;
    image_resized.convertTo(fimage, CV_32FC1);

    cv::Scalar mean(127.5);
    fimage -= mean;

    for (int row = 0; row < fimage.rows; ++row) {
      const float *fimage_begin = fimage.ptr<const float>(row);
      const float *fimage_end = fimage_begin + fimage.cols;
      std::copy(fimage_begin, fimage_end,
                input + row * fimage.cols + i * fimage.cols * fimage.rows);
    }

    // read label
    std::vector<std::string> indices_str_v;
    split(indices_str, ',', &indices_str_v);
    std::vector<int64_t> indices;
    for (auto index_str : indices_str_v) {
      indices.push_back(std::stol(index_str));
    }
    gt_label.push_back(indices);
  }
  return true;
}
} // namespace paddle
