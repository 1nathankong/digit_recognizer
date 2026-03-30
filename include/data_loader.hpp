#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mnist
{
    struct Dataset {
        std::vector<float> images; // [60000 * 28 * 28]
        std::vector<float> labels; // [60000]
        int count = 0;
    };

    Dataset load_dataset(const std::string& image_path, const std::string& label_path); // image, label, just like defined in the struct
}