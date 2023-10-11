#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <filesystem>

int main() {
    //Input image filepaths
    std::string image1_path;
    std::cout << "Enter path for first image: ";
    std::cin >> image1_path;

    std::string image2_path;
    std::cout << "Enter path for second image: ";
    std::cin >> image2_path;

    //Check if filepaths exist
    if (!std::filesystem::exists(image1_path) || !std::filesystem::exists(image2_path)) {
        std::cout << "Filepath invalid" << std::endl;
        return -1;
    }
    //Load image from paths
    cv::Mat
    imageA = cv::imread(image1_path),
    imageB = cv::imread((image2_path));

    //Check if valid image file
    if (imageA.empty() || imageB.empty()){
        std::cout << "Not valid image files";
        return -1;
    }
    else{
        std::cout <<"Images loaded successfully" << std::endl;
    }
    //Input image alpha values
    double alpha;
    std::cout << "Enter the alpha [0.0 - 1.0]: ";
    std::cin >> alpha;

    //Check if they are the same size
    if (imageA.size >= imageB.size)
    {
        //Resize A to B size
        cv::resize(imageA,imageA, imageB.size());
    }
    else
    {
        //Resize B to A size
        cv::resize(imageB,imageB, imageA.size());
    }

    //Show original images
    cv::imshow("ImageA", imageA);
    cv::imshow("ImageB", imageB);
    cv::waitKey(0);
    cv::destroyAllWindows();

    cv::Mat tempImage;

    //Matrix cycle to sum pixel values  NON-FUNCTIONAL
    /*for (int y = 0; y <= imageA.rows; y++) {
        for (int x = 0; x <= imageA.cols; x++) {
            for (int c = 0; c < imageA.channels(); ++c) {
                tempImage.row(y) = alpha * imageA.row(y) + (1.0 - alpha) * imageB.row(y);
                tempImage.col(x) = alpha * imageA.col(x) + (1.0 - alpha) * imageB.col(x);
            }
        }
    }*/

    //Show modified image
    /*cv::imshow("tempImage", tempImage);
    cv::waitKey(0);
    cv::destroyAllWindows();*/

    //Non cycle sum
    tempImage = alpha * imageA + (1.0 - alpha) * imageB;

    //Show modified image
    cv::imshow("tempImage", tempImage);
    cv::waitKey(0);
    cv::destroyAllWindows();
}
