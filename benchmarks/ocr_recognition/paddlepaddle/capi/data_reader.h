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

// Data reader for imagenet for ResNet50
struct DataReader {
  explicit DataReader(const std::string &data_list_path,
                      const std::string &data_dir_path, int image_width,
                      int image_height, int channels);

  bool NextBatch(float *input, std::vector<std::vector<int64_t>> &gt_label,
                 int batch_size);

private:
  std::string data_list_path;
  std::string data_dir_path;
  std::ifstream file;
  int image_width;
  int image_height;
  int channels;
  char sep{' '};
};
} // namespace paddle
