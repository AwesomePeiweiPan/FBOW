
//First step of creating a vocabulary is extracting features from a set of images. We save them to a file for next step
#include <iostream>
#include <fstream>
#include <vector>
#include "fbow.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif

using namespace fbow;
using namespace std;


//command line parser
class CmdLineParser{int argc; char **argv; public: CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}  bool operator[] ( string param ) {int idx=-1;  for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i;    return ( idx!=-1 ) ;    } string operator()(string param,string defvalue="-1"){int idx=-1;    for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue;   else  return ( argv[  idx+1] ); }};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void wait()
{
    cout << endl << "Press enter to continue" << endl;
    getchar();
}


vector<string> readImagePathsFromFile(char * txtlist){
    vector<string> paths;
    // Open file
	std::ifstream infile(txtlist, std::ios::binary);

    while(!infile.eof())
    {
        string line;
        getline(infile, line);

        if (line.length() > 0)
           paths.push_back(line);

        //std::cout << line << std::endl;
    }

    return paths;
}

vector< cv::Mat  >  loadFeatures( std::vector<string> path_to_images,string descriptor="") {
    //select detector
    cv::Ptr<cv::Feature2D> fdetector;
    if (descriptor=="orb")        fdetector=cv::ORB::create(2000);
    else if (descriptor=="brisk") fdetector=cv::BRISK::create();
#ifdef OPENCV_VERSION_3
    else if (descriptor=="akaze") fdetector=cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB,  0,  3, 1e-4);
#endif
#ifdef USE_CONTRIB
    else if(descriptor=="surf" )  fdetector=cv::xfeatures2d::SURF::create(15, 4, 2);
#endif

    else throw std::runtime_error("Invalid descriptor");
    assert(!descriptor.empty());
    vector<cv::Mat>    features;


    cout << "Extracting   features..." << endl;
    for(size_t i = 0; i < path_to_images.size(); ++i)
    {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        cout<<"reading image: "<<path_to_images[i]<<endl;
        cv::Mat image = cv::imread(path_to_images[i], 0);
        if(image.empty())throw std::runtime_error("Could not open image"+path_to_images[i]);
        cout<<"extracting features"<<endl;
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        features.push_back(descriptors);
        cout<<"done detecting features (" << i << "/" << path_to_images.size() << ")" <<endl;
    }
    return features;
}

// ----------------------------------------------------------------------------
void saveToFile(string filename,const vector<cv::Mat> &features,  std::string  desc_name,bool rewrite =true){

    //test it is not created
    if (!rewrite){
        std::fstream ifile(filename);
        if (ifile.is_open())//read size and rewrite
            std::runtime_error( "ERROR::: Output File "+filename+" already exists!!!!!" );
    }
	std::ofstream ofile(filename, std::ios::binary);
    if (!ofile.is_open()){cerr<<"could not open output file"<<endl;exit(0);}

    char _desc_name[20];
    desc_name.resize(min(size_t(19),desc_name.size()));
    strcpy(_desc_name,desc_name.c_str());
    ofile.write(_desc_name,20);

    uint32_t size=features.size();
    ofile.write((char*)&size,sizeof(size));
    for(auto &f:features){
        if( !f.isContinuous()){
            cerr<<"Matrices should be continuous"<<endl;exit(0);
        }
        uint32_t aux=f.cols; ofile.write( (char*)&aux,sizeof(aux));
        aux=f.rows; ofile.write( (char*)&aux,sizeof(aux));
        aux=f.type(); ofile.write( (char*)&aux,sizeof(aux));
        ofile.write( (char*)f.ptr<uchar>(0),f.total()*f.elemSize());
    }
}

// ----------------------------------------------------------------------------
//代码主要用于从给定的图片列表中提取特征，并将这些特征保存到文件中。它允许用户选择不同的特征提取方法，并使用OpenCV库来完成特征提取的工作。
int main(int argc,char **argv)
{

    try{
        CmdLineParser cml(argc,argv);
        if (cml["-h"] || argc==1){
            cerr<<"Usage:  descriptor_name output txtlist \n\t descriptors:brisk,surf,orb(default),akaze(only if using opencv 3)"<<endl;
            return -1;
        }

        string descriptor=argv[1];
        string output=argv[2];

        auto images=readImagePathsFromFile(argv[3]);
        vector< cv::Mat   >   features= loadFeatures(images,descriptor);

        //save features to file
        saveToFile(argv[2],features,descriptor);

    }catch(std::exception &ex){
        cerr<<ex.what()<<endl;
    }

    return 0;
}
