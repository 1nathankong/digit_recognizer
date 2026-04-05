#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <fstream>

struct Dataset {
    std::vector<float> images;
    std::vector<uint8_t> labels;
};

void parse_labels(std::ifstream& file);
void parse_images(std::ifstream& file);
void define_dataset(std::ifstream& img_file, std::ifstream&lbl_file, Dataset& data);
void load(const std::string& path, Dataset& data);