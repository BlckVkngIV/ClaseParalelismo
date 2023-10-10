#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <filesystem>

     int main() {
         //Input image file path
         std::string image_path;
         std::cout << "Enter the path to the image: ";
         std::cin >> image_path;
         //Check if filepath exists
         if (!std::filesystem::exists(image_path)) {
             std::cout << "File does not exist at the specified path" << std::endl;
             return -1;
         }

         cv::Mat image = cv::imread(image_path);
         //Check if its a valid image file
         if (image.empty()) {
             std::cout << "Error loading the image" << std::endl;
             return -1;
         }
         else {
             std::cout << "Image loaded successfully" << std::endl;
         }
         //Show original image
         cv::imshow("Image", image);

         //Wait for a keystroke in the window
         cv::waitKey(0);

         // Separate the image into its three channels
         cv::Mat bgr[3];
         cv::split(image, bgr);

         //Modify images by color maps
         cv::Mat blueChannel, greenChannel, redChannel;
         cv::applyColorMap(bgr[0], blueChannel, cv::COLORMAP_JET);
         cv::applyColorMap(bgr[1], greenChannel, cv::COLORMAP_SPRING);
         cv::applyColorMap(bgr[2], redChannel, cv::COLORMAP_AUTUMN);

         // Create modified image windows
         cv::imshow("blueChannel", blueChannel);
         cv::imshow("greenChannel", greenChannel);
         cv::imshow("redChannel", redChannel);
         cv::waitKey(0);
     }
