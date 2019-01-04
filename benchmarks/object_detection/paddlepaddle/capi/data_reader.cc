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
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <fstream>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <sys/stat.h>

using boost::property_tree::ptree;
using boost::property_tree::read_xml;

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
                       const std::string &data_dir_path,
                       const std::string &label_list_path, int resize_size,
                       int channels, bool convert_to_rgb)
    : data_list_path(data_list_path), data_dir_path(data_dir_path),
      label_list_path(label_list_path), file(data_list_path),
      resize_size(resize_size), channels(channels),
      convert_to_rgb(convert_to_rgb) {
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

  if (channels != 3) {
    throw std::invalid_argument("Only 3 channel image loading supported");
  }

  InitLabelMap();
}

void DataReader::InitLabelMap() {
  int i = 0;
  std::string label;
  std::ifstream label_file{label_list_path};
  if (!label_file.is_open()) {
    throw std::invalid_argument("Cannot open label list file " +
                                label_list_path);
  }
  while (std::getline(label_file, label)) {
    boost::trim(label);
    label_map[label] = i;
    ++i;
  }
}

void DataReader::drawImages(float *input, bool is_rgb, int batch_size,
                            int channels, int width, int height) {
  for (int b = 0; b < batch_size; b++) {
    std::vector<cv::Mat> fimage_channels;
    for (int c = 0; c < channels; c++) {
      fimage_channels.emplace_back(cv::Size(width, height), CV_32FC1,
                                   input + width * height * c +
                                       width * height * channels * b);
    }
    cv::Mat mat;
    if (is_rgb) {
      std::swap(fimage_channels[0], fimage_channels[2]);
    }
    cv::merge(fimage_channels, mat);
    cv::imshow(std::to_string(b) + " output image", mat);
  }
  std::cout << "Press any key in image window or close it to continue"
            << std::endl;
  cv::waitKey(0);
}

// return true if separator works or false otherwise
bool DataReader::SetSeparator(char separator) {
  sep = separator;

  std::string line;
  auto position = file.tellg();
  std::getline(file, line);
  file.clear();
  file.seekg(position);

  // test out
  std::vector<std::string> pieces;
  split(line, separator, &pieces);

  return (pieces.size() == 2);
}

bool DataReader::ReadOneXMLToBox(const std::string &annotationfile,
                                 std::vector<int64_t> &gt_label,
                                 std::vector<std::vector<float>> &gt_bbox,
                                 std::vector<int64_t> &gt_difficult) {
  ptree tree;
  read_xml(annotationfile, tree);

  int height = tree.get("annotation.size.height", 0);
  int width = tree.get("annotation.size.width", 0);

  if (height == 0 || width == 0) {
    std::stringstream ss;
    ss << "You read out height = " << height << "and width is  " << width
       << "Unexpected 0 size";
    throw std::runtime_error(ss.str());
  }
  for (ptree::value_type &content_annotation : tree.get_child("annotation")) {
    if (content_annotation.first == "object") {
      ptree object_tree = content_annotation.second;
      for (ptree::value_type &content_object : object_tree.get_child("")) {
        ptree pt2 = content_object.second;
        if (content_object.first == "name") {
          std::string name = pt2.data();
          if (label_map.find(name) == label_map.end()) {
            throw std::runtime_error(
                "Find some name that is not in the label list");
          }
          int64_t labelindex = label_map.find(name)->second;
          // tmpOnebbox.insert(it,labelindex);
          gt_label.push_back(labelindex);
        }
        if (content_object.first == "bndbox") {
          int xmin = pt2.get("xmin", 0);
          int ymin = pt2.get("ymin", 0);
          int xmax = pt2.get("xmax", 0);
          int ymax = pt2.get("ymax", 0);
          float bbox1 = static_cast<float>(xmin) / width;
          float bbox2 = static_cast<float>(ymin) / height;
          float bbox3 = static_cast<float>(xmax) / width;
          float bbox4 = static_cast<float>(ymax) / height;
          gt_bbox.push_back({bbox1, bbox2, bbox3, bbox4});
        }
        if (content_object.first == "difficult") {
          int64_t difficult = 0;
          if (pt2.data() == "1") {
            difficult = 1;
          }
          gt_difficult.push_back(difficult);
          // tmpOnebbox.insert(it+5,difficult);
        }
      }
    }
  }
}

bool DataReader::NextBatch(
    float *input, std::vector<std::vector<int64_t>> &gt_label,
    std::vector<std::vector<std::vector<float>>> &gt_bbox,
    std::vector<std::vector<int64_t>> &gt_difficult, int batch_size,
    bool debug_display_images) {
  gt_label.clear();
  gt_bbox.clear();
  gt_difficult.clear();
  gt_label.resize(batch_size);
  gt_bbox.resize(batch_size);
  gt_difficult.resize(batch_size);

  std::string line;
  for (int i = 0; i < batch_size; i++) {
    if (!std::getline(file, line))
      return false;

    std::vector<std::string> pieces;
    split(line, sep, &pieces);
    if (pieces.size() != 2) {
      std::stringstream ss;
      ss << "invalid number of separators '" << sep << "' found in line " << i
         << ":'" << line << "' of file " << data_list_path;
      throw std::runtime_error(ss.str());
    }

    auto filename = data_dir_path + pieces.at(0);
    auto annotationfile = data_dir_path + pieces.at(1);

    ReadOneXMLToBox(annotationfile, gt_label[i], gt_bbox[i], gt_difficult[i]);

    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);

    if (image.data == nullptr) {
      std::string error_msg = "Couldn't open file " + filename;
      throw std::runtime_error(error_msg);
    }

    if (convert_to_rgb) {
      cv::cvtColor(image, image, CV_BGR2RGB);
    }

    if (debug_display_images)
      cv::imshow(std::to_string(i) + " input image", image);

    cv::Mat image_resized = resize(image, resize_size, resize_size);

    cv::Mat fimage;
    image_resized.convertTo(fimage, CV_32FC3);

    // for channels BGR
    cv::Scalar mean(127.5, 127.5, 127.5);
    cv::Scalar scale(0.007843f, 0.007843f, 0.007843f);

    if (convert_to_rgb) {
      std::swap(mean[0], mean[2]);
      std::swap(scale[0], scale[2]);
    }

    std::vector<cv::Mat> fimage_channels;
    cv::split(fimage, fimage_channels);

    for (int c = 0; c < channels; c++) {
      fimage_channels[c] -= mean[c];
      fimage_channels[c] *= scale[c];
      for (int row = 0; row < fimage.rows; ++row) {
        const float *fimage_begin = fimage_channels[c].ptr<const float>(row);
        const float *fimage_end = fimage_begin + fimage.cols;
        std::copy(fimage_begin, fimage_end,
                  input + row * fimage.cols + c * fimage.cols * fimage.rows +
                      i * 3 * fimage.cols * fimage.rows);
      }
    }
  }
  return true;
}

} // namespace paddle
