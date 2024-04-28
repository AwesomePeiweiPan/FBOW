#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <map>
#include <mutex> 
#include <algorithm>
#include <regex>

using namespace std;
namespace fs = std::filesystem;



std::vector<std::tuple<double, int, int, double, int>> parameters = {
    {0.02, 5, 1, 5, 1},
    {0.01, 5, 1, 5, 1},
    //{0.005, 5, 1, 5, 1},
    {0.01, 5, 1, 5, 1},
    //{0.02, 5, 1, 5, 1},
    {0.01, 10, 1, 10, 1}
};

//分数高于多少才被认为有共视；默认0.02，自动化，如果没有合适的结果，会降低这个值
//double thresholdScore = 0.01; 
//在Images0中至少连续多少张图片才被认定为有效图片 默认5
//const int ConsecutiveImages = 5;
//在Images0中相邻照片序数相差多少才合适 默认3
//const int neighborThresholdImages0=1;
//在Images1中每个组至少要有多少张照片 默认4
//const double thresholdSecondConseutiveImages = 5;
//在Images1中相邻照片序数相差多少才合适 默认3
//const int neighborThreshold = 1;
//////基本的文件路径
//地图匹配结果
std::string GroupSequence = "/home/peiweipan/fbow/Euroc_Data_more/GroupSequence/GroupSequence.txt"; 
//Scores所在的文件夹地址
std::string ScoresFiles = "/home/peiweipan/fbow/Euroc_Data_more/"; 
//关键帧源文件
std::string cam0_Images = "/home/peiweipan/Projects/DroidSlam/Euroc_Data/KeyFrames_more/cam0"; 
std::string cam1_Images = "/home/peiweipan/Projects/DroidSlam/Euroc_Data/KeyFrames_more/cam1"; 
//输出回环的文件夹
fs::path loop_Output = "/home/peiweipan/Projects/DroidSlam/Euroc_Data/Loop_more";





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


std::string findScoreFolder(const fs::path& folder, int num1, int num2) {
    for (const auto& entry : fs::directory_iterator(folder)) {
        const auto& filename = entry.path().filename().string();
        if (filename.find(std::to_string(num1)) != std::string::npos && 
            filename.find(std::to_string(num2)) != std::string::npos) {
            return entry.path().string();
        }
    }
    return "";
}

std::vector<fs::path> findFoldersInCamFolders(const fs::path& cam0_folder, 
                                              const fs::path& cam1_folder, 
                                              int num1, int num2) {
    std::vector<fs::path> folders(4); // 创建一个包含4个元素的向量

    // 在 cam0 中按照 num1, num2 的顺序查找并存储路径
    for (const auto& entry : fs::directory_iterator(cam0_folder)) {
        const auto& filename = entry.path().filename().string();
        if (filename.find(std::to_string(num1)) != std::string::npos) {
            folders[0] = entry.path(); // 存储 num1 对应的 cam0 文件夹路径
        } else if (filename.find(std::to_string(num2)) != std::string::npos) {
            folders[1] = entry.path(); // 存储 num2 对应的 cam0 文件夹路径
        }
    }

    // 在 cam1 中按照 num1, num2 的顺序查找并存储路径
    for (const auto& entry : fs::directory_iterator(cam1_folder)) {
        const auto& filename = entry.path().filename().string();
        if (filename.find(std::to_string(num1)) != std::string::npos) {
            folders[2] = entry.path(); // 存储 num1 对应的 cam1 文件夹路径
        } else if (filename.find(std::to_string(num2)) != std::string::npos) {
            folders[3] = entry.path(); // 存储 num2 对应的 cam1 文件夹路径
        }
    }

    return folders;
}


fs::path createLoopFolder(const fs::path& parentDir, int num1, int num2) {
    fs::path targetFolder = parentDir / (std::to_string(num1) + "_" + std::to_string(num2));

    if (fs::exists(targetFolder)) {
        fs::remove_all(targetFolder);
    }
    fs::create_directories(targetFolder);

    return targetFolder; // 返回新创建的文件夹的路径
}




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

void transformScores(std::vector<std::map<int, double>>& scores) {
    if (scores.empty()) {
        return; // 如果原始 scores 为空，则直接返回
    }

    // 确定原始 scores 的维度
    int outerSize = scores.size();
    int innerSize = scores[0].size(); // 假设所有 map 的大小相同

    // 创建新的数据结构
    std::vector<std::map<int, double>> newScores(innerSize);

    // 重组数据
    for (int i = 0; i < outerSize; ++i) {
        for (const auto& pair : scores[i]) {
            int newKey = i; // 新的 map 键是原始 vector 的索引
            int newIndex = pair.first; // 新的 vector 索引是原始 map 的键
            newScores[newIndex][newKey] = pair.second;
        }
    }

    // 更新原始 scores
    scores = std::move(newScores);
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
void processGroup(const std::vector<std::map<int, double>>& scores, GroupedSequence& gSeq, 
                  std::vector<int>& currentGroup, int B) {
    int a = gSeq.firstSet.back();
    int b = currentGroup[0];
    int c = currentGroup.back();
    std::string order;
    if (scores.at(a).at(b) > scores.at(a).at(c)) {
        order = "Order";
    } else {
        order = "ReverseOrder";
    }

    if (order == "Order") {
        while (currentGroup.back() < B) {
            currentGroup.push_back(currentGroup.back() + 1);
        }
    } else {
        while (currentGroup.front() > 0) {
            currentGroup.insert(currentGroup.begin(), currentGroup.front() - 1);
        }
    }

    gSeq.groupedSecondSets.push_back(currentGroup);
    gSeq.orders.push_back(order);

    std::vector<int> newGroup;
    if (order == "Order") {
        for (int i = 0; i < b; ++i) {
            newGroup.push_back(i);
        }
        gSeq.orders.push_back("ReverseOrder");
    } else {
        for (int i = c + 1; i < B+1; ++i) {   //
            newGroup.push_back(i);
        }
        gSeq.orders.push_back("Order");
    }

    if (!newGroup.empty()) {
        gSeq.groupedSecondSets.push_back(newGroup);
    }
}





std::vector<GroupedSequence> groupSecondSets(const std::vector<Sequence>& sequences,
                                             const std::vector<std::map<int, double>>& scores, 
                                             int B, const double thresholdSecondConseutiveImages, const int neighborThreshold) {
    int N2 = scores.empty() ? 0 : scores[0].size();

    std::vector<GroupedSequence> groupedSequences;

    for (const auto& seq : sequences) {
        GroupedSequence gSeq;
            // 如果seq.firstSet的元素数量大于40，只取最后40个元素
        if (seq.firstSet.size() > 40) {
            gSeq.firstSet.assign(seq.firstSet.end() - 40, seq.firstSet.end());
        } else {
            gSeq.firstSet = seq.firstSet;
        }

        while (gSeq.firstSet.size() < 40) {
            int nextNumber = gSeq.firstSet.front() - 1;
            if (nextNumber < 0) {
                break;
            }
            gSeq.firstSet.insert(gSeq.firstSet.begin(), nextNumber);
        }

        std::vector<int> currentGroup;
        bool isGroupStart = true;
        for (size_t i = 0; i < seq.secondSet.size(); ++i) {
            if (isGroupStart || abs(seq.secondSet[i] - seq.secondSet[i - 1]) <= neighborThreshold) {
                currentGroup.push_back(seq.secondSet[i]);
                isGroupStart = false;
            } else {
                if (currentGroup.size() > thresholdSecondConseutiveImages) {
                    processGroup(scores, gSeq, currentGroup, B);
                }
                currentGroup.clear();
                currentGroup.push_back(seq.secondSet[i]);
                isGroupStart = false;
            }
        }

        // 处理循环结束后的最后一组
        if (!currentGroup.empty() && currentGroup.size() > thresholdSecondConseutiveImages) {
            processGroup(scores, gSeq, currentGroup, B);
        }

        if (!gSeq.groupedSecondSets.empty()) {
            groupedSequences.push_back(gSeq);
        }
    }

    return groupedSequences;
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

void clear_directory(const fs::path &dir) {
    for (const auto &entry : fs::directory_iterator(dir)) {
        fs::remove_all(entry.path());
    }
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

void writeGroupedSequencesToFile(const std::vector<GroupedSequence>& groupedSequences, const fs::path& folderPath) {
    fs::path filePath = folderPath / "Sequence.txt";  // 构造文件路径

    // 确保目标文件夹存在
    if (!fs::exists(folderPath)) {
        fs::create_directories(folderPath);  // 如果文件夹不存在，则创建
    }

    std::ofstream outfile(filePath, std::ios::out | std::ios::trunc);

    if (!outfile.is_open()) {
        std::cerr << "Failed to open the file: " << filePath << std::endl;
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

bool isFileEmpty(const fs::path& folderPath) {
    fs::path filePath = folderPath / "Sequence.txt"; // 构造文件的完整路径

    if (!fs::exists(filePath)) {
        // 如果文件不存在，可以认为是“空”的
        return true;
    }

    std::ifstream file(filePath);
    return file.peek() == std::ifstream::traits_type::eof();
}

int getMaxKeyFromScores(const std::vector<std::map<int, double>>& scores) {
    int maxKey = std::numeric_limits<int>::min(); // 初始设为int的最小值

    for (const auto& scoreMap : scores) {
        if (!scoreMap.empty()) {
            int localMax = std::max_element(scoreMap.begin(), scoreMap.end(),
                                            [](const auto& a, const auto& b) {
                                                return a.first < b.first;
                                            })->first;
            maxKey = std::max(maxKey, localMax);
        }
    }

    return maxKey;
}

int main() {
    std::ifstream file(GroupSequence);
    std::string line;
    int iterationCount = 0; // 计数器

    while (std::getline(file, line)) {
        auto [thresholdScore, ConsecutiveImages, neighborThresholdImages0, thresholdSecondConseutiveImages, neighborThreshold] = parameters[iterationCount % parameters.size()];

        int num1, num2;
        std::istringstream iss(line);
        if (!(iss >> num1 >> num2)) { break; } // 错误或行结束
        std::string PairsScores = findScoreFolder(ScoresFiles, num1, num2);
        std::vector<std::map<int, double>> scores;
        readFromTxt(PairsScores, scores);
        if(num1 > num2){ transformScores(scores);}

        //选取每组数据最大的20个double
        std::vector<std::map<int, double>> MAXScores;

        for (const auto& oldMap : scores) {
            auto topN = getTopN(oldMap, 5); // 提取最大的20个
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

        //创建文件夹
        double originalThresholdScore = thresholdScore; 
        do {
            std::vector<SubData> extractedData;
            // 使用std::copy_if函数将满足条件的元素复制到新的vector中
            std::copy_if(BestMatch.begin(), BestMatch.end(), std::back_inserter(extractedData),
                        [&originalThresholdScore](const SubData& data) { return data.averageValue > originalThresholdScore; });

            //提取第0组照片中连续超过阈值数量的图片序列
            std::vector<std::vector<SubData>> allConsecutiveData = extractGroupedData(extractedData, ConsecutiveImages, neighborThresholdImages0);
    
            //合并结果
            std::vector<Sequence> sequences = transformToSequence(allConsecutiveData);

            //保证第1组照片中，每组子数据是满足要求的子序列，比如相邻数据之间的绝对值小于等于6，每组至少有10个数据
            int maxKey = getMaxKeyFromScores(scores);
            std::vector<GroupedSequence> BestSequences = groupSecondSets(sequences,scores,maxKey,thresholdSecondConseutiveImages, neighborThreshold);

            auto Key_images_folders = findFoldersInCamFolders(cam0_Images, cam1_Images, num1, num2);
            fs::path outputFolder = createLoopFolder(loop_Output, num1, num2);

            createAndCopyFolders(Key_images_folders[0], Key_images_folders[1], Key_images_folders[2], Key_images_folders[3], outputFolder, BestSequences);
            writeGroupedSequencesToFile(BestSequences, outputFolder);

            // 如果文件为空，则调整thresholdScore并重复循环
            if (isFileEmpty(outputFolder)) {
                originalThresholdScore *= 0.9;
            } else {
                break;
            }
        }while (true);
        iterationCount++; // 增加计数器
    }

    return 0;
}
