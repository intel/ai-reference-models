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

#pragma once

#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace paddle {

struct DataReader {
  explicit DataReader(const std::string &data_list_path,
                      const std::string &data_dir_path,
                      const std::string &label_list_path, int resize_size,
                      int channels, bool convert_to_rgb);

  static void drawImages(float *input, bool is_rgb, int batch_size,
                         int channels, int width, int height);

  // return true if separator works or false otherwise
  bool SetSeparator(char separator);

  bool
  DataReader::NextBatch(float *input,
                        std::vector<std::vector<int64_t>> &gt_label,
                        std::vector<std::vector<std::vector<float>>> &gt_bbox,
                        std::vector<std::vector<int64_t>> &gt_difficult,
                        int batch_size, bool debug_display_images);

private:
  bool ReadOneXMLToBox(const std::string &annotationfile,
                       std::vector<int64_t> &gt_label,
                       std::vector<std::vector<float>> &gt_bbox,
                       std::vector<int64_t> &gt_difficult);
  void InitLabelMap();
  std::string data_list_path;
  std::string data_dir_path;
  std::string label_list_path;
  std::ifstream file;
  std::map<std::string, int> label_map;
  int resize_size;
  int channels;
  bool convert_to_rgb;
  char sep{'\t'};
};

} // namespace paddle
