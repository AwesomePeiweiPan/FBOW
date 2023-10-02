#include <fbow/fbow.h>
#include <iostream>
#include <filesystem>
#include <map>
#include <fstream>
#include <thread>  // for std::thread
#include <mutex> 
#include <algorithm>

using namespace std;

struct SubData {
    int imagesNums1;
    double averageValue;
    std::vector<int> imagesNums2;

    SubData(int idx, double avg) : imagesNums1(idx), averageValue(avg) {}
};

std::vector<std::map<double, int>> readFromTxt(const std::string& filename, size_t startLine, size_t endLine) {
    std::vector<std::map<double, int>> result;
    std::ifstream in(filename);
    std::string line;

    size_t lineNumber = 0;
    while (std::getline(in, line)) {
        if (lineNumber >= startLine && lineNumber < endLine) {
            std::map<double, int> m;
            std::stringstream ss(line);
            std::string pair;
            while (ss >> pair) {
                size_t commaPos = pair.find(',');
                double key = std::stod(pair.substr(0, commaPos));
                int value = std::stoi(pair.substr(commaPos + 1));
                m[key] = value;
            }
            result.push_back(m);
        }
        lineNumber++;
    }
    return result;
}

bool compareMaps(const std::map<double, int>& a, const std::map<double, int>& b) {
    // 这里假设 map 不为空，若可能为空需要增加额外的判断
    return a.begin()->second < b.begin()->second;
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

class CmdLineParser { int argc; char **argv; public: CmdLineParser(int _argc, char **_argv) :argc(_argc), argv(_argv) {}  bool operator[] (string param) { int idx = -1;  for (int i = 0; i<argc && idx == -1; i++) if (string(argv[i]) == param) idx = i;    return (idx != -1); } string operator()(string param, string defvalue = "-1") { int idx = -1;    for (int i = 0; i<argc && idx == -1; i++) if (string(argv[i]) == param) idx = i; if (idx == -1) return defvalue;   else  return (argv[idx + 1]); } };

int main(int argc, char **argv){

    CmdLineParser cml(argc, argv);
    if (argc<2 || cml["-h"]) throw std::runtime_error("Usage: Lines numThreads");

    //多线程读取文件
    const int numThreads = std::stoi(argv[2]);
    std::vector<std::vector<std::map<double, int>>> threadData(numThreads);

    size_t totalLines = std::stoi(argv[1]);  // 假设你知道或已经计算了文件的行数
    size_t blockSize = totalLines / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        size_t start = i * blockSize;
        size_t end = (i == numThreads - 1) ? totalLines : (i + 1) * blockSize;
        threadData[i] = readFromTxt("/home/peiweipan/fbow/ImageMatch/scores.txt", start, end);
    }

    std::vector<std::map<double, int>> scores;
    for (auto& data : threadData) {
        scores.insert(scores.end(), data.begin(), data.end());
    }

    //改变键值对的排列，从double int -> int double
    std::vector<std::map<int, double>> newscores;

    for (const auto& oldMap : scores) {
        std::map<int, double> newMap;
        for (const auto& [key, value] : oldMap) {
            newMap[value] = key;
        }
        newscores.push_back(newMap);
        cout<< "Changing key value sequence" << endl;
    }
    
    //选取每组数据最大的20个double
    std::vector<std::map<int, double>> MAXScores;

    for (const auto& oldMap : newscores) {
        auto topN = getTopN(oldMap, 20); // 提取最大的20个
        std::map<int, double> newMap(topN.begin(), topN.end());
        MAXScores.push_back(newMap);
        cout<< "Getiing max 20" << endl;
    }

     std::vector<SubData> results;
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
        results.push_back(subData);
        cout<< "Calculating average values" << endl;
    }

    //写入文件
    writeToFile(results, "/home/peiweipan/fbow/ImageMatch/BestMatch.txt");

    return 0;
}