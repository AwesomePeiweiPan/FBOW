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
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif


#include <chrono>
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

std::mutex mtx;  // Global mutex for scores
std::mutex scores_mutex;

void processImage(size_t i, fbow::Vocabulary& voc, const vector<vector<cv::Mat>>& features1, const vector<vector<cv::Mat>>& features2, vector<map<double, int>>& scores)
{
    fbow::fBow vv = voc.transform(features1[i][0]);
    map<double, int> score;
    for (size_t j = 0; j < features2.size(); ++j)
    {
        fbow::fBow vv2 = voc.transform(features2[j][0]);
        double score1 = vv.score(vv, vv2);
        score.insert(pair<double, int>(score1, j));
    }
    mtx.lock();  // Locking the mutex
    scores[i] = score;
    mtx.unlock();  // Releasing the mutex
}

void loadFeaturesForRange1(const std::vector<string>& filenames, const string& desc_name, int startIdx, int endIdx, vector<vector<cv::Mat>>& features) {
    for (size_t i = startIdx; i < endIdx && i < filenames.size(); ++i) {
        features[i] = loadFeatures({ filenames[i] }, desc_name);
    }
}

void computeScoresForRange(const vector<vector<cv::Mat>>& features1, const vector<vector<cv::Mat>>& features2, fbow::Vocabulary& voc, int startIdx, int endIdx, vector<map<double, int>>& scores) {
    for (size_t i = startIdx; i < endIdx && i < features1.size(); ++i) {
        fbow::fBow vv = voc.transform(features1[i][0]);
        map<double, int> score;
        for (size_t j = 0; j < features2.size(); ++j) {
            fbow::fBow vv2 = voc.transform(features2[j][0]);
            double score1 = vv.score(vv, vv2);
            score.insert(pair<double, int>(score1, j));
            cout<<i<<" , "<<j<<endl;
        }
        scores_mutex.lock();
        scores[i] = score;
        scores_mutex.unlock();
    }
}

std::mutex file_mutex; 
void writeToTxt(const std::vector<std::map<double, int>>& data, const std::string& filename, size_t start, size_t end) {
    std::ofstream out;

    // 将锁定和文件操作放在同一范围内，确保写入操作的完整性
    file_mutex.lock();
    out.open(filename, std::ios::app); // 以追加模式打开

    for (size_t i = start; i < end; ++i) {
        for (auto const& [key, value] : data[i]) {
            out << key << "," << value << " ";
        }
        out << std::endl;
        cout<< "Writing for i" <<endl;
    }
    
    out.close();
    file_mutex.unlock();
}

int countLinesInFile(const std::string& filename) {
    std::ifstream in(filename);
    return std::count(std::istreambuf_iterator<char>(in),
                      std::istreambuf_iterator<char>(), '\n');
}


int main(int argc, char **argv) {
    CmdLineParser cml(argc, argv);
    try {
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
        int TimeCost = 0;
        
        auto t_start = std::chrono::high_resolution_clock::now();
        


        vector<vector<cv::Mat> > features1(SizeImages1);
        vector<vector<cv::Mat> > features2(SizeImages2);
        vector<map<double, int> > scores;
        scores.resize(features1.size());
        fbow::fBow vv, vv2;

        const int numThreads = 2;
        std::vector<std::thread> threads1;
        int blockSize1 = SizeImages1 / numThreads;
        for (int t = 0; t < numThreads; ++t) {
            threads1.push_back(std::thread(loadFeaturesForRange1, std::ref(filenames1), desc_name, t * blockSize1, (t + 1) * blockSize1, std::ref(features1)));
        }
        for (auto& th : threads1) {
            th.join();
        }

        std::vector<std::thread> threads2;
        int blockSize2 = SizeImages2 / numThreads;
        for (int t = 0; t < numThreads; ++t) {
            threads2.push_back(std::thread(loadFeaturesForRange1, std::ref(filenames2), desc_name, t * blockSize2, (t + 1) * blockSize2, std::ref(features2)));
        }
        for (auto& th : threads2) {
            th.join();
        }

        //const int numThreads2=14;
        const int numThreads2=14;
        std::vector<std::thread> threads3;
        int blockSize3 = features1.size() / numThreads2;
        for (int t = 0; t < numThreads2; ++t) {
            threads3.push_back(std::thread(computeScoresForRange, std::ref(features1), std::ref(features2), std::ref(voc), t * blockSize3, (t + 1) * blockSize3, std::ref(scores)));
        }
        for (auto& th : threads3) {
            th.join();
        }


        auto t_end = std::chrono::high_resolution_clock::now();
        TimeCost = double(std::chrono::duration_cast<std::chrono::seconds>(t_end - t_start).count()) / 60.0;
        cout << "TimeCost: " << TimeCost <<endl;

        std::ofstream clearFile("/home/peiweipan/fbow/ImageMatch/scores.txt", std::ios::out); // 这会清空文件内容
        clearFile.close();

        std::vector<std::thread> threads;
         size_t blockSize = scores.size() / numThreads2;
         for (int i = 0; i < numThreads2; ++i) {
        size_t start = i * blockSize;
        size_t end = (i == numThreads2 - 1) ? scores.size() : (i + 1) * blockSize;
        threads.push_back(std::thread(writeToTxt, std::ref(scores), "/home/peiweipan/fbow/ImageMatch/scores.txt", start, end));
        }
            for (auto& t : threads) {
                t.join();
        }

        int lineCount = countLinesInFile("/home/peiweipan/fbow/ImageMatch/scores.txt");
        std::cout << "The file has " << lineCount << " lines." << std::endl;

    }
    catch (std::exception &ex) {
        cerr << ex.what() << endl;
    }

}