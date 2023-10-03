//loads a vocabulary, and a image. Extracts image feaures and then  compute the bow of the image
#include <fbow/fbow.h>
#include <iostream>
#include <filesystem>
#include <map>
#include <fstream>

#include <thread>  // for std::thread
#include <mutex> 

using namespace std;

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

class CmdLineParser { int argc; char **argv; public: CmdLineParser(int _argc, char **_argv) :argc(_argc), argv(_argv) {}  bool operator[] (string param) { int idx = -1;  for (int i = 0; i<argc && idx == -1; i++) if (string(argv[i]) == param) idx = i;    return (idx != -1); } string operator()(string param, string defvalue = "-1") { int idx = -1;    for (int i = 0; i<argc && idx == -1; i++) if (string(argv[i]) == param) idx = i; if (idx == -1) return defvalue;   else  return (argv[idx + 1]); } };



vector< cv::Mat  >  loadFeatures(std::vector<string> path_to_images, string descriptor = "")  {
    //select detector
    cv::Ptr<cv::Feature2D> fdetector;
    if (descriptor == "orb")        fdetector = cv::ORB::create(2000);
    else if (descriptor == "brisk") fdetector = cv::BRISK::create();
#ifdef OPENCV_VERSION_3
    else if (descriptor == "akaze") fdetector = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 1e-4);
#endif
#ifdef USE_CONTRIB
    else if (descriptor == "surf")  fdetector = cv::xfeatures2d::SURF::create(15, 4, 2);
#endif

    else throw std::runtime_error("Invalid descriptor");
    assert(!descriptor.empty());
    vector<cv::Mat>  features;


    cout << "Extracting   features..." << endl;
    for (size_t i = 0; i < path_to_images.size(); ++i)
    {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        cout << "reading image: " << path_to_images[i] << endl;
        cv::Mat image = cv::imread(path_to_images[i], 0);
        if (image.empty())throw std::runtime_error("Could not open image" + path_to_images[i]);
        cout << "extracting features" << endl;
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        features.push_back(descriptors);
        cout << "done detecting features" << endl;
    }
    return features;
}


int main(int argc, char **argv) {
    CmdLineParser cml(argc, argv);
    if (argc<3 || cml["-h"]) throw std::runtime_error("Usage: fbow path_to_imagefile1 path_to_imagefile2");
    fbow::Vocabulary voc;
    voc.readFromFile(argv[1]);

    string desc_name = voc.getDescName();
    cout << "voc desc name=" << desc_name << endl;

    string path_to_imagefile1 = argv[2];
    string path_to_imagefile2 = argv[3];

    vector<string> filenames1;
        for (const auto& entry : filesystem::directory_iterator(path_to_imagefile1)) {
            if (entry.is_regular_file()) {
                 filenames1.push_back(entry.path().string());
            }
        }
        std::sort(filenames1.begin(), filenames1.end());
        if (filenames1.empty()) { cerr << "Path_to_imagefile1 contains no files. Exiting..." << endl; return 1;}
        int SizeImages1 = filenames1.size();
        cout<< "imagefile1 contains" << SizeImages1 << "images" << endl;
    
    vector<string> filenames2;
        for (const auto& entry : filesystem::directory_iterator(path_to_imagefile2)) {
            if (entry.is_regular_file()) {
                 filenames2.push_back(entry.path().string());
            }
        }
        std::sort(filenames2.begin(), filenames2.end());
        if (filenames2.empty()) {cerr << "Path_to_imagefile2 contains no files. Exiting..." << endl; return 1; }
        int SizeImages2 = filenames2.size();
        cout<< "imagefile2 contains" << SizeImages2 << "images" << endl;

    const int i = 48;
    const int j =87;

    vector<vector<cv::Mat> > features1(1);
    vector<vector<cv::Mat> > features2(1);
    features1[0] = loadFeatures({ filenames1[i] }, desc_name);
    features2[0] = loadFeatures({ filenames2[j] }, desc_name);

    fbow::fBow vv, vv2;

    vv = voc.transform(features1[0][0]);
    vv2 = voc.transform(features2[0][0]);

    double score1 = vv.score(vv, vv2);
    cout<<score1<<endl;

    return 0;
}