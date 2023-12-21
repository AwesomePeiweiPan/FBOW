#include <fbow/fbow.h>
#include <iostream>
#include <filesystem>
#include <map>
#include <fstream>
#include <thread>  // for std::thread
#include <mutex> 
#include <chrono>
#include <algorithm>
#include <string>
#include <vector>
#include <regex>
#include <queue>
#include <set>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif

using namespace std;
namespace fs = std::filesystem;

//////需要修改的参数
//voc文件路径
string vocabularyPath = "/home/peiweipan/fbow/Pairs_Hilti/Voc/vocHilti.fbow";
//string vocabularyPath = "/home/peiweipan/fbow/Pairs/Voc/voc.fbow";

//图片源
vector<string> imageSequenceFolders = {
    "/home/peiweipan/Projects/DroidSlam/HiltiData/KeyframesInte/cam0/handheld01",
    "/home/peiweipan/Projects/DroidSlam/HiltiData/KeyframesInte/cam0/handheld02",
    "/home/peiweipan/Projects/DroidSlam/HiltiData/KeyframesInte/cam0/handheld03",
    "/home/peiweipan/Projects/DroidSlam/HiltiData/KeyframesInte/cam0/handheld04",
};
//得到分数后，需要输出的地方
string outputScoresPath = "/home/peiweipan/fbow/Hilti_Maps";
//增加线程提高效率
const int numThreads = 24;
//在第一次运行的时候设置为false，以后设置为true，大大提高运行速度
bool scoresFileHasBeenSaved = false;
//设置合适的阈值，保证每对图片组都有一定的关联性
const double relationScoresThreads = 0.005;
//设置从哪个地图开始
const int OriginalStartMap = 1;
//将匹配好的图片组对输出到一个地方
string outputGroupSequencePath = "/home/peiweipan/fbow/Hilti_Maps/GroupSequence/GroupSequence.txt"; 
//////








////// 以下是第一大部分，计算分数部分需要的函数
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

    for (size_t i = 0; i < path_to_images.size(); ++i)
    {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        cv::Mat image = cv::imread(path_to_images[i], 0);
        if (image.empty())throw std::runtime_error("Could not open image" + path_to_images[i]);
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        features.push_back(descriptors);
    }
    return features;
}

std::mutex mtx;  // Global mutex for scores
std::mutex scores_mutex;

void loadFeaturesForRange1(const std::vector<string>& filenames, const string& desc_name, int startIdx, int endIdx, vector<vector<cv::Mat>>& features) {
    for (size_t i = startIdx; i < endIdx && i < filenames.size(); ++i) {
        features[i] = loadFeatures({ filenames[i] }, desc_name);
    }
}

void computeScoresForRange(const vector<vector<cv::Mat>>& features1, const vector<vector<cv::Mat>>& features2, fbow::Vocabulary& voc, int startIdx, int endIdx, vector<map<int, double>>& scores) {
    for (size_t i = startIdx; i < endIdx && i < features1.size(); ++i) {
        fbow::fBow vv = voc.transform(features1[i][0]);
        map<int, double> score;
        for (size_t j = 0; j < features2.size(); ++j) {
            fbow::fBow vv2 = voc.transform(features2[j][0]);
            double score1 = vv.score(vv, vv2);
            score.insert(pair<int, double>(j, score1));
        }
        scores_mutex.lock();
        scores[i] = score;
        scores_mutex.unlock();
    }
}

void writeToTxt(const std::vector<std::map<int, double>>& data, const std::string& filename, size_t start, size_t end) {
    // 从文件名中提取目录路径
    std::filesystem::path filePath(filename);
    std::filesystem::path dirPath = filePath.parent_path();

    // 检查目录是否存在，如果不存在则创建
    if (!std::filesystem::exists(dirPath) && !dirPath.empty()) {
        std::filesystem::create_directories(dirPath);
    }
    
    std::ofstream out;
    out.open(filename, std::ios::out | std::ios::trunc); // 使用截断模式打开文件

    if (!out.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }

    for (size_t i = start; i < end; ++i) {
        for (auto const& [key, value] : data[i]) {
            out << key << "," << value << " ";
        }
        out << std::endl;
    }
    
    out.close();
}

////// 以下是第二大部分，给出图像匹配结果的结构和函数
struct GroupScore {
    int num1;
    int num2;
    int countAboveThreshold;
};

struct SubData {
    int imagesNums1;
    double averageValue;
    std::vector<int> imagesNums2;

    SubData(int idx, double avg) : imagesNums1(idx), averageValue(avg) {}
};
std::vector<SubData> BestMatch;

struct Edge {
    int v1, v2;
    int weight;
    bool operator<(const Edge& other) const {
        return weight < other.weight; // 改为比较权重较小的边
    }
};

bool readFromTxt(const std::string& filename, std::vector<std::map<int, double>>& data) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::map<int, double> mapData;

        std::string kv;
        while (iss >> kv) {
            size_t pos = kv.find(",");
            if (pos != std::string::npos) {
                int key = std::stoll(kv.substr(0, pos));
                double value = std::stod(kv.substr(pos + 1));
                mapData[key] = value;
            } else {
                std::cerr << "Invalid key-value pair format: " << kv << std::endl;
            }
        }

        data.push_back(mapData);
    }

    in.close();
    return true;
}

std::vector<std::pair<int, double>> getTopN(const std::map<int, double>& m, int N) {
    std::vector<std::pair<int, double>> result(N);
    std::partial_sort_copy(
        m.begin(), m.end(),
        result.begin(), result.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.second > rhs.second;
        });
    return result;
}

bool extractNumbers(const string& filename, int& num1, int& num2) {
    std::regex reg("(\\d+).*?(\\d+)");
    std::smatch matches;
    if (std::regex_search(filename, matches, reg) && matches.size() == 3) {
        num1 = std::stoi(matches[1].str());
        num2 = std::stoi(matches[2].str());
        return true;
    }
    return false;
}

std::vector<std::vector<int>> extractDataFromMST(const std::vector<Edge>& mst) {
    std::vector<std::vector<int>> result;
    for (const Edge& edge : mst) {
        result.push_back({edge.v1+1, edge.v2+1});
    }
    return result;
}

std::vector<Edge> applyPrimAlgorithmForMaxTree(const std::vector<Edge>& edges, int totalNodes) {
    std::vector<Edge> maxTree;
    std::set<int> selectedNodes;
    std::priority_queue<Edge> edgeQueue; // 默认为最大堆

    // 假设起始节点是0（或者选择任何一个节点作为起始节点）
    selectedNodes.insert(OriginalStartMap-1);
    // 初始化边的优先队列4
    std::map<int, std::vector<Edge>> adjList;
    for (const auto& edge : edges) {
        adjList[edge.v1].push_back(edge);
        adjList[edge.v2].push_back({edge.v2, edge.v1, edge.weight}); // 对于无向图
    }

    for (const auto& edge : adjList[OriginalStartMap-1]) {
        edgeQueue.push(edge);
    }

    while (!edgeQueue.empty() && selectedNodes.size() < totalNodes) {
        Edge maxEdge = edgeQueue.top();
        edgeQueue.pop();

        if (selectedNodes.find(maxEdge.v2) != selectedNodes.end()) continue; // 跳过已选节点

        maxTree.push_back(maxEdge); // 加入最大生成树
        selectedNodes.insert(maxEdge.v2); // 标记节点

        for (const auto& edge : adjList[maxEdge.v2]) {
            if (selectedNodes.find(edge.v2) == selectedNodes.end()) {
                edgeQueue.push(edge);
            }
        }
    }

    return maxTree;
}

void savePairsToFile(const std::vector<std::vector<int>>& vec, const std::string& filename) {
    // 从文件名中提取目录路径
    std::filesystem::path filePath(filename);
    std::filesystem::path dirPath = filePath.parent_path();

    // 检查目录是否存在，如果不存在则创建
    if (!std::filesystem::exists(dirPath) && !dirPath.empty()) {
        std::filesystem::create_directories(dirPath);
    }

    std::ofstream outFile(filename, std::ios::out | std::ios::trunc); // 使用 trunc 模式打开文件
    if (!outFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (const auto& innerVec : vec) {
        for (const auto& item : innerVec) {
            outFile << item << " ";
        }
        outFile << std::endl;
    }

    outFile.close();
}


int main() {
    if (!scoresFileHasBeenSaved) {
        fbow::Vocabulary voc;
        voc.readFromFile(vocabularyPath);
        string desc_name = voc.getDescName();

        for (size_t i = 0; i < imageSequenceFolders.size(); ++i) {
            for (size_t j = i + 1; j < imageSequenceFolders.size(); ++j) {
                string path_to_imagefile1 = imageSequenceFolders[i];
                string path_to_imagefile2 = imageSequenceFolders[j];
                
                //第一步：从文件夹中读取照片
                vector<string> filenames1;
                for (const auto& entry : filesystem::directory_iterator(path_to_imagefile1)) {
                    if (entry.is_regular_file()) {
                    filenames1.push_back(entry.path().string());
                }
                }
                std::sort(filenames1.begin(), filenames1.end());
                if (filenames1.empty()) { cerr << "Path_to_imagefile1 contains no files. Exiting..." << endl; return 1;}
                int SizeImages1 = filenames1.size();
                cout<< "Image Pairs" << i << "," << j << endl;
                cout<< "First Images Series contains" << SizeImages1 << "images" << endl;

                vector<string> filenames2;
                for (const auto& entry : filesystem::directory_iterator(path_to_imagefile2)) {
                    if (entry.is_regular_file()) {
                        filenames2.push_back(entry.path().string());
                    }
                }
                std::sort(filenames2.begin(), filenames2.end());
                if (filenames2.empty()) {cerr << "Path_to_imagefile2 contains no files. Exiting..." << endl; return 1; }
                int SizeImages2 = filenames2.size();
                cout<< "Second Images Series contains" << SizeImages2 << "images" << endl;
                int TimeCost = 0;
        
                auto t_start = std::chrono::high_resolution_clock::now();
        
                vector<vector<cv::Mat> > features1(SizeImages1);
                vector<vector<cv::Mat> > features2(SizeImages2);
                vector<map<int, double> > scores;
                scores.resize(features1.size());
                fbow::fBow vv, vv2;

                //第二步：从图片中提取特征
                cout<<"Extract Features"<<endl;
                std::vector<std::thread> threads1;
                int blockSize1 = SizeImages1 / (numThreads-1);
                for (int t = 0; t < numThreads; ++t) {
                    int startIdx = t * blockSize1;
                    // 如果是最后一个线程，确保它处理所有剩余的图片
                    int endIdx = (t == numThreads - 1) ? SizeImages1 : (t + 1) * blockSize1;
                    threads1.push_back(std::thread(loadFeaturesForRange1, std::ref(filenames1), desc_name, startIdx, endIdx, std::ref(features1)));
                }
                for (auto& th : threads1) {
                    th.join();
                }

                std::vector<std::thread> threads2;
                int blockSize2 = SizeImages2 / (numThreads-1);
                for (int t = 0; t < numThreads; ++t) {
                    int startIdx = t * blockSize2;
                    // 如果是最后一个线程，确保它处理所有剩余的图片
                    int endIdx = (t == numThreads - 1) ? SizeImages2 : (t + 1) * blockSize2;
                    threads2.push_back(std::thread(loadFeaturesForRange1, std::ref(filenames2), desc_name, startIdx, endIdx, std::ref(features2)));
                }
                for (auto& th : threads2) {
                    th.join();
                }

                //第三步：根据提取的特征进行计算相关性分数
                cout<<"Compute Scores"<<endl;
                std::vector<std::thread> threads3;
                int blockSize3 = features1.size() / (numThreads-1);
                for (int t = 0; t < numThreads; ++t) {
                    int startIdx = t * blockSize3;
                    // 如果是最后一个线程，确保它处理所有剩余的图片
                    int endIdx = (t == numThreads - 1) ? features1.size() : (t + 1) * blockSize3;
                    threads3.push_back(std::thread(computeScoresForRange, std::ref(features1), std::ref(features2), std::ref(voc), startIdx, endIdx, std::ref(scores)));
                }
                for (auto& th : threads3) {
                    th.join();
                }
                auto t_end = std::chrono::high_resolution_clock::now(); 
                TimeCost = double(std::chrono::duration_cast<std::chrono::seconds>(t_end - t_start).count());
                cout << "TimeCost: " << TimeCost <<endl;  
                //第四步：存储结果到一个文件夹中
                string filename = outputScoresPath + "/" +
                                  filesystem::path(path_to_imagefile1).filename().string() +
                                  filesystem::path(path_to_imagefile2).filename().string() +
                                  ".txt";
                std::ofstream clearFile(filename, std::ios::out);
                clearFile.close();                  
                writeToTxt(scores, filename, 0, scores.size());
                cout <<"Finish Calculating Scores for"<< i << "," << j <<endl;
            }
        }
    }
    //第五步：循环读取Scores并且计算相关性
    vector<GroupScore> allGroupScore;
    for (const auto& entry : fs::directory_iterator(outputScoresPath)) {
        if (entry.path().extension() == ".txt") {
            string filePath = entry.path().string();
            vector<map<int, double>> scores;
            readFromTxt(filePath, scores);
        std::vector<std::map<int, double>> MAXScores;
        for (const auto& oldMap : scores) {
            auto topN = getTopN(oldMap, 10); // 提取最大的20个
            std::map<int, double> newMap(topN.begin(), topN.end());
            MAXScores.push_back(newMap);
        }

        std::vector<SubData> BestMatch;
        //计算每组数据的平均值
        for (int i = 0; i < MAXScores.size(); ++i) {
            const auto& m = MAXScores[i];
            double sum = 0;
            for (const auto& [key, value] : m) {
                sum += value;
            }
            double avg = m.empty() ? 0 : sum / m.size();
            SubData subData(i, avg);
            for (const auto& [key, value] : m) {
                subData.imagesNums2.push_back(key);
            }
            BestMatch.push_back(subData);
        }
            int num1, num2;
            if (extractNumbers(entry.path().stem().string(), num1, num2)) {
                // 如果成功提取了数字
                int countAboveThreshold = 0;
                for (const SubData& data : BestMatch) {
                    if (data.averageValue > relationScoresThreads) {
                        countAboveThreshold++;
                    }
                }

                allGroupScore.push_back({num1, num2, countAboveThreshold});
            }
        }
    }
    //第六步：使用Prim最大生成树进行构建，得到符合的结果
    std::vector<Edge> edges; 
    for (const auto& result : allGroupScore) {
        edges.push_back({result.num1-1, result.num2-1, result.countAboveThreshold});
    }
    int TotalMaps = imageSequenceFolders.size();
    std::vector<Edge> mst = applyPrimAlgorithmForMaxTree(edges, TotalMaps);
    std::vector<std::vector<int>> edgePairs = extractDataFromMST(mst);
    //第七步：存储结果
    savePairsToFile(edgePairs, outputGroupSequencePath);
    return 0;
    

}
