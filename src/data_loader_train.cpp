#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <bit>
#include <string>

//  --- Global Variables ---
constexpr int OFFSET = 4; //4 byte offset
constexpr int TRAIN_IMAGES = 60000; //total number of images
constexpr int IMG_ROWS = 28; // 28 pixels 
constexpr int IMG_COLS = 28; // 28 pixels
constexpr int IMG_SIZE = IMG_ROWS * IMG_COLS; // 28 * 28 to get size of pixels dimensions



struct Dataset {
    std::vector<float> images;
    std::vector<uint8_t> labels;
    
    void resize(size_t num_items)
    {
        labels.resize(num_items);
        images.resize(num_items * 784);
    }
};



void parse_labels(std::ifstream& file)
{
    int32_t magic;
    int32_t items;
    
    file.read(reinterpret_cast<char*>(&magic), OFFSET);
    magic = std::byteswap(magic);
    
    file.read(reinterpret_cast<char*>(&items), OFFSET);
    items = std::byteswap(items);

}

void parse_images(std::ifstream& file)
{
    int32_t magic_number;
    int32_t num_images;
    int32_t num_rows;
    int32_t num_columns;

    file.read(reinterpret_cast<char*>(&magic_number), OFFSET);
    magic_number = std::byteswap(magic_number);

    file.read(reinterpret_cast<char*>(&num_images), OFFSET);
    num_images = std::byteswap(num_images);
    
    file.read(reinterpret_cast<char*>(&num_rows), OFFSET);
    num_rows = std::byteswap(num_rows);
    file.read(reinterpret_cast<char*>(&num_columns), OFFSET);
    num_columns = std::byteswap(num_columns);
}

void define_dataset(std::ifstream& img_file, std::ifstream&lbl_file, Dataset& data)
{
    data.resize(TRAIN_IMAGES);
    //data.images.resize(TRAIN_IMAGES*IMG_SIZE);
    
    lbl_file.read(reinterpret_cast<char*>(data.labels.data()), TRAIN_IMAGES);
    
    std::vector<uint8_t> temp_pixels(TRAIN_IMAGES*IMG_SIZE);
    img_file.read(reinterpret_cast<char*>(temp_pixels.data()), temp_pixels.size());
    for(size_t i = 0; i < temp_pixels.size(); i++)
    {
        data.images[i] = static_cast<float>(temp_pixels[i])/ 255.0f;
    }
}

void load(const std::string& path, Dataset& data)
{
    std::string train_image = path + "train-images.idx3-ubyte";
    std::string train_labels = path + "train-labels.idx1-ubyte";

    std::ifstream image_file(train_image, std::ios::binary);
    std::ifstream label_file(train_labels, std::ios::binary);

    if(image_file.is_open() && label_file.is_open()){
        parse_images(image_file);
        parse_labels(label_file);
        define_dataset(image_file, label_file, data);
        std::cout << "dataset can be parsed and defined" << std::endl;
    }
}


void verify_data(const Dataset& data, int index) {
    if (index >= data.labels.size()) {
        std::cout << "Index out of bounds!" << std::endl;
        return;
    }

    std::cout << "\n--- Verifying Index " << index << " ---" << std::endl;
    std::cout << "Label: " << (int)data.labels[index] << std::endl;

    // Calculate where this specific image starts in the flat vector
    int start_pos = index * 784; 

    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
            // Get pixel using the offset for this specific image
            float pixel = data.images[start_pos + (r * 28 + c)];
            std::cout << (pixel > 0.5f ? "##" : "  ");
        }
        std::cout << std::endl;
    }
}


int main() {
    Dataset myData;
    load("../", myData);

    for (int i = 0; i < 100; i++) {
        verify_data(myData, i);
    }

    return 0;
}

