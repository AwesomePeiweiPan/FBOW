#include <fbow/fbow.h>
#include <iostream>
#include <filesystem>
#include <map>
#include <fstream>
#include <thread>  // for std::thread
#include <mutex> 
#include <algorithm>
#include <string>
#include <vector>
#include <regex>


using namespace std;
namespace fs = std::filesystem;

//设定阈值 
//在Images0中至少连续多少张图片才被认定为有效图片
const int ConsecutiveImages = 5;
//在Images0中相邻照片序数相差多少才合适
const int neighborThresholdImages0=3;
//分数高于多少才被认为有共视；
const double thresholdScore = 0.02; 
//在Images1中每个组至少要有多少张照片
const double thresholdSecondConseutiveImages = 4;
//在Images1中相邻照片序数相差多少才合适
const int neighborThreshold = 3;

struct SubData {
    int imagesNums1;
    double averageValue;
    std::vector<int> imagesNums2;

    SubData(int idx, double avg) : imagesNums1(idx), averageValue(avg) {}
};

struct Sequence {
    std::vector<int> firstSet;
    std::vector<int> secondSet;
};

struct GroupedSequence {
    std::vector<int> firstSet;  // 从原始Sequence继承
    std::vector<std::vector<int>> groupedSecondSets;  // 存储符合条件的子序列
    std::vector<std::string> orders;
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

// 将SubData写入txt文件
void writeToFile(const std::vector<SubData>& data, const std::string& filename) {
    std::ofstream outfile(filename, std::ios::out | std::ios::trunc); // 添加 trunc 标志
    for (const auto& item : data) {
        outfile << item.imagesNums1 << "," << item.averageValue;
        for (int key : item.imagesNums2) {
            outfile << "," << key;
        }
        outfile << std::endl;
        cout<< "Writing Results" << endl;
    }
    outfile.close();
}

std::vector<std::vector<SubData>> extractGroupedData(const std::vector<SubData>& extractedData, const int& ConsecutiveImages, const int& threshold) {
    std::vector<SubData> sortedData = extractedData;

    // 根据imagesNums1排序sortedData
    std::sort(sortedData.begin(), sortedData.end(),
              [](const SubData& a, const SubData& b) { return a.imagesNums1 < b.imagesNums1; });

    std::vector<std::vector<SubData>> allGroupedData;

    for (size_t i = 0; i < sortedData.size();) {
        int start = i;
        int count = 1;

        // 寻找符合条件的子vector
        while (i + 1 < sortedData.size() && 
               (sortedData[i + 1].imagesNums1 - sortedData[i].imagesNums1) <= threshold) {
            ++i;
            ++count;
        }

        if (count >= ConsecutiveImages) {
            std::vector<SubData> group(sortedData.begin() + start, sortedData.begin() + start + count);
            allGroupedData.push_back(group);
            i += count;
        } else {
            ++i;
        }
    }

    return allGroupedData;
}

std::vector<Sequence> transformToSequence(const std::vector<std::vector<SubData>>& allConsecutiveData) {
    std::vector<Sequence> sequences;
    for (const auto& group : allConsecutiveData) {
        Sequence seq;
        for (const auto& data : group) {
            seq.firstSet.push_back(data.imagesNums1);
            seq.secondSet.insert(seq.secondSet.end(), data.imagesNums2.begin(), data.imagesNums2.end());
        }
        // 对secondSet进行排序
        std::sort(seq.secondSet.begin(), seq.secondSet.end());
        // 使用unique和erase去重
        seq.secondSet.erase(std::unique(seq.secondSet.begin(), seq.secondSet.end()), seq.secondSet.end());
        sequences.push_back(seq);
    }
    return sequences;
}

std::vector<GroupedSequence> groupSecondSets(const std::vector<Sequence>& sequences,
                                             const std::vector<std::map<int, double>>& scores) {
    std::vector<GroupedSequence> groupedSequences;

    for (const auto& seq : sequences) {
        GroupedSequence gSeq;
        gSeq.firstSet = seq.firstSet;
        int a = gSeq.firstSet.back();  // firstSet的最后一个值

        std::vector<int> currentGroup;
        for (size_t i = 1; i < seq.secondSet.size(); ++i) {
            if (abs(seq.secondSet[i] - seq.secondSet[i-1]) <= neighborThreshold) {
                currentGroup.push_back(seq.secondSet[i-1]);
                if (i == seq.secondSet.size() - 1) {
                    currentGroup.push_back(seq.secondSet[i]);
                    if (currentGroup.size() > thresholdSecondConseutiveImages) {
                        gSeq.groupedSecondSets.push_back(currentGroup);
                        int b = currentGroup[0];
                        int c = currentGroup.back();
                        if (scores.at(a).at(b) > scores.at(a).at(c)) {
                            gSeq.orders.push_back("Order");
                        } else {
                            gSeq.orders.push_back("ReverseOrder");
                        }
                        currentGroup.clear();
                    }
                }
            } else {
                if (currentGroup.size() > thresholdSecondConseutiveImages) {
                    gSeq.groupedSecondSets.push_back(currentGroup);
                    int b = currentGroup[0];
                    int c = currentGroup.back();
                    if (scores.at(a).at(b) > scores.at(a).at(c)) {
                        gSeq.orders.push_back("Order");
                    } else {
                        gSeq.orders.push_back("ReverseOrder");
                    }
                    
                }
                currentGroup.clear();
            }
        }
        
        if (!gSeq.groupedSecondSets.empty()) {
            groupedSequences.push_back(gSeq);
        }
    }
    
    return groupedSequences;
}


void writeGroupedSequencesToFile(const std::vector<GroupedSequence>& groupedSequences, const std::string& filename) {
    std::ofstream outfile(filename, std::ios::out | std::ios::trunc);

    if (!outfile.is_open()) {
        std::cerr << "Failed to open the file: " << filename << std::endl;
        return;
    }

    int index = 0;
    for (const auto& gSeq : groupedSequences) {
        outfile << index << std::endl;
        outfile << "First Images Series: ";
        for (const auto& value : gSeq.firstSet) {
            outfile << value << " ";
        }
        outfile << std::endl;

        int subIndex = 0;
        for (const auto& set : gSeq.groupedSecondSets) {
            outfile << "Second Images Series" << subIndex << ": ";
            for (const auto& value : set) {
                outfile << value << " ";
            }
            outfile << " [" << gSeq.orders[subIndex] << "]";  // Append the order information
            outfile << std::endl;
            ++subIndex;
        }
        ++index;
    }

    outfile.close();
}

std::vector<std::string> getSortedFiles(const fs::path& path) {
    std::regex r(R"((\d+).png)");
    std::vector<std::pair<long long, std::string>> numberedFiles;

    for (const auto& entry : fs::directory_iterator(path)) {
        std::string filePath = entry.path().string();
        std::smatch match;
        if (std::regex_search(filePath, match, r)) {
            numberedFiles.emplace_back(std::stoll(match[1].str()), entry.path().string());
        }
    }

    std::sort(numberedFiles.begin(), numberedFiles.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    std::vector<std::string> sortedFiles;
    for (const auto& pair : numberedFiles) {
        sortedFiles.push_back(pair.second);
    }

    return sortedFiles;
}

int copyFiles(const std::vector<std::string>& sourceFiles, const std::vector<int>& indices,
              const std::string& destDir, int currentPrefix, bool reverse = false) {
    if (reverse) {
        for (auto it = indices.rbegin(); it != indices.rend(); ++it) {
            fs::path sourcePath = sourceFiles[*it];
            fs::path destDirPath(destDir);
            fs::path destFilePath = destDirPath / (std::to_string(currentPrefix) + "_" + sourcePath.filename().string());
            fs::copy(sourcePath, destFilePath, fs::copy_options::overwrite_existing);
            ++currentPrefix;
        }
    } else {
        for (auto it = indices.begin(); it != indices.end(); ++it) {
            fs::path sourcePath = sourceFiles[*it];
            fs::path destDirPath(destDir);
            fs::path destFilePath = destDirPath / (std::to_string(currentPrefix) + "_" + sourcePath.filename().string());
            fs::copy(sourcePath, destFilePath, fs::copy_options::overwrite_existing);
            ++currentPrefix;
        }
    }
    return currentPrefix;  // 返回最后的prefix，便于下次调用
}


void clear_directory(const fs::path &dir) {
    for (const auto &entry : fs::directory_iterator(dir)) {
        fs::remove_all(entry.path());
    }
}

// 新增函数来处理cam0和cam1的文件夹，避免重复代码
void handleCameraFolder(const GroupedSequence& seq, const fs::path& camPath, 
                        const std::vector<std::string>& sortedFiles1, 
                        const std::vector<std::string>& sortedFiles2) {

    int subDirIdx = 0;
    for (size_t i = 0; i < seq.groupedSecondSets.size(); i++) {
        fs::path subDirPath = camPath / ("sub" + std::to_string(subDirIdx));
        fs::create_directories(subDirPath);

        // Copy from first path
        int currentPrefix = copyFiles(sortedFiles1, seq.firstSet, subDirPath.string(), 1);
        // Copy from second path
        if (seq.orders[i] == "ReverseOrder") {
            copyFiles(sortedFiles2, seq.groupedSecondSets[i], subDirPath.string(), currentPrefix, true);
        } else if (seq.orders[i] == "Order") {
            copyFiles(sortedFiles2, seq.groupedSecondSets[i], subDirPath.string(), currentPrefix);
        }

        subDirIdx++;
    }
}

void createAndCopyFolders(const std::string& path1, const std::string& path2, 
                          const std::string& path3, const std::string& path4, // 新增path3和path4
                          const std::string& destBasePath,
                          const std::vector<GroupedSequence>& BestSequences) {

    // 在运行函数前，清空destBasePath路径下的所有内容，并删除该路径
    if (fs::exists(destBasePath)) {
        fs::remove_all(destBasePath);
    }
    fs::create_directories(destBasePath);  // 重新创建destBasePath目录

    auto sortedFiles1 = getSortedFiles(path1);
    auto sortedFiles2 = getSortedFiles(path2);
    auto sortedFiles3 = getSortedFiles(path3); // 新增sortedFiles3
    auto sortedFiles4 = getSortedFiles(path4); // 新增sortedFiles4

    int sequenceDirIdx = 0; 

    for (const auto& seq : BestSequences) {
        fs::path currentSeqPath = fs::path(destBasePath) / std::to_string(sequenceDirIdx);
        
        // 处理cam0
        fs::path cam0Path = currentSeqPath / "cam0";
        if (fs::exists(cam0Path)) {
            clear_directory(cam0Path);
        }
        fs::create_directories(cam0Path);
        handleCameraFolder(seq, cam0Path, sortedFiles1, sortedFiles2);

        // 处理cam1
        fs::path cam1Path = currentSeqPath / "cam1"; // 新的cam1路径
        if (fs::exists(cam1Path)) {
            clear_directory(cam1Path);
        }
        fs::create_directories(cam1Path);
        handleCameraFolder(seq, cam1Path, sortedFiles3, sortedFiles4); // 使用sortedFiles3和sortedFiles4

        sequenceDirIdx++;
    }
}








int main(int argc, char **argv){

    std::vector<std::map<int, double>> scores;
    readFromTxt("/home/peiweipan/fbow/Euroc_MH/scores.txt", scores);
    
    
    //选取每组数据最大的20个double
     std::vector<std::map<int, double>> MAXScores;

    for (const auto& oldMap : scores) {
        auto topN = getTopN(oldMap, 20); // 提取最大的20个
        std::map<int, double> newMap(topN.begin(), topN.end());
        MAXScores.push_back(newMap);
        cout<< "Getiing max 20" << endl;
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
        cout<< "Calculating average values" << endl;
    }

    //写入文件
    writeToFile(BestMatch, "/home/peiweipan/fbow/Euroc_MH/BestMatch.txt");


    //统计
    std::vector<std::vector<SubData>> categorizedData(12);
    for (const SubData& data : BestMatch) {
        if (data.averageValue < 0.01) {
            categorizedData[0].push_back(data);
        } else if (data.averageValue < 0.1) {
            // 注意这里我们使用了static_cast<int>(data.averageValue * 100) - 1
            // 例如，如果averageValue为0.03，则这会得到2，它指向categorizedData的第三个位置
            categorizedData[static_cast<int>(data.averageValue * 100) - 1].push_back(data);
        } else {
            categorizedData[11].push_back(data);
        }
    }

    // 输出每个子数组中的元素个数
    for (int i = 0; i < categorizedData.size(); ++i) {
        if (i == 0) {
            std::cout << "Count of elements with averageValue < 0.01: " << categorizedData[i].size() << std::endl;
        } else if (i <= 9) {
            std::cout << "Count of elements with averageValue in (" << i * 0.01 << ", " << (i + 1) * 0.01 << "]: " << categorizedData[i].size() << std::endl;
        } else {
            std::cout << "Count of elements with averageValue > 0.1: " << categorizedData[i].size() << std::endl;
        }
    }

    std::vector<SubData> extractedData;


    // 使用std::copy_if函数将满足条件的元素复制到新的vector中
    std::copy_if(BestMatch.begin(), BestMatch.end(), std::back_inserter(extractedData),
                 [](const SubData& data) { return data.averageValue > thresholdScore; });

    //提取第0组照片中连续超过阈值数量的图片序列
    std::vector<std::vector<SubData>> allConsecutiveData = extractGroupedData(extractedData, ConsecutiveImages, neighborThresholdImages0);
    
    //合并结果
    std::vector<Sequence> sequences = transformToSequence(allConsecutiveData);

    //保证第1组照片中，每组子数据是满足要求的子序列，比如相邻数据之间的绝对值小于等于6，每组至少有10个数据
    std::vector<GroupedSequence> BestSequences = groupSecondSets(sequences,scores);

    writeGroupedSequencesToFile(BestSequences, "/home/peiweipan/fbow/Euroc_MH/Sequence.txt");

    std::string outputFolder = "/home/peiweipan/fbow/Euroc_MH/KeyFrames/MH01_02_Loop";  // 替换为你的输出文件夹路径
    const string MH01_cam0 = "/home/peiweipan/fbow/Euroc_MH/KeyFrames/MH01_cam0";
    const string MH02_cam0 = "/home/peiweipan/fbow/Euroc_MH/KeyFrames/MH02_cam0";
    const string MH01_cam1 = "/home/peiweipan/fbow/Euroc_MH/KeyFrames/MH01_cam1";
    const string MH02_cam1 = "/home/peiweipan/fbow/Euroc_MH/KeyFrames/MH02_cam1";


    createAndCopyFolders(MH01_cam0, MH02_cam0, MH01_cam1, MH02_cam1, outputFolder, BestSequences);





    return 0;
}