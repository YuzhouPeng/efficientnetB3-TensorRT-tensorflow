#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "math.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 0;

// image resizing var
int imgh = 224;
int imgw = 224;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
// char s[6] = {"a","b","c","d","e","f"};
using namespace nvinfer1;

static Logger gLogger;

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}




IScaleLayer* addBatchNorm2d(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps)
{
    float* gamma = (float*)weightMap[lname + "gamma:0"].values; // scale
    float* beta = (float*)weightMap[lname + "beta:0"].values;   // offset
    float* mean = (float*)weightMap[lname + "moving_mean:0"].values;
    float* var = (float*)weightMap[lname + "moving_variance:0"].values;
    int len = weightMap[lname + "moving_variance:0"].count;
    std::cout << "len " << len << std::endl;
    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (auto i = 0; i < len; i++)
    {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (auto i = 0; i < len; i++)
    {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (auto i = 0; i < len; i++)
    {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;



}

IConvolutionLayer* addSamePaddingConv(INetworkDefinition *network, std::map<std::string, Weights>& weightMap,ITensor& input,int imgheight,int imgwidth,int kernelsize, int stride,int outch,int groups,Weights convweights,Weights bias){
    int kw = kernelsize;
    int kh = kernelsize;
    int iw = imgwidth;
    int ih = imgheight;
    int sh = stride;
    int sw = stride;
    int oh = ceil(ih/sh);
    int ow = ceil(iw/sw);
    int padh = std::max((oh-1)*sh+(kh-1)+1-ih,0);
    int padw = std::max((ow-1)*sw+(kw-1)+1-iw,0);
    // IPoolingLayer* pool1;
    int pad_top = 0;
    int pad_bottom = 0;
    int pad_left = 0;
    int pad_right = 0;
    if (padw>0 || padh>0){
        pad_left = int(padw/2);
        pad_right = padw-int(padw/2);
        pad_top = int(padh/2);
        pad_bottom = padh-int(padh/2);
    }
    std::cout << "padw and padh "<< padw<<"  "<<padh<< std::endl;
    std::cout << "groups  "<< groups<<"  "<< std::endl;
    IConvolutionLayer* samepaddingconv = network->addConvolutionNd(input, outch, DimsHW{kernelsize, kernelsize}, convweights, bias);
    samepaddingconv->setStrideNd(DimsHW{stride, stride});
    // samepaddingconv->setPaddingNd(DimsHW{0, 0});

    samepaddingconv->setPrePadding(DimsHW{pad_top, pad_left});
    samepaddingconv->setPostPadding(DimsHW(pad_bottom,pad_right));
    samepaddingconv->setStrideNd(DimsHW{stride, stride});
    samepaddingconv->setNbGroups(groups);
    // samepaddingconv->setDilationNd(Dims{1,1});
    Dims diminput1 = samepaddingconv->getOutput(0)->getDimensions();
    int channels = diminput1.d[0];
    int windowsizeh1 = diminput1.d[1];
    int windowsizew1 = diminput1.d[2];
    std::cout << "same padding output layer size "<< windowsizeh1<<"  "<<windowsizew1<<" "<<channels<< std::endl;

    
    
    return samepaddingconv;

}

int calculateoutputimgheight(int inputh,int stride){
    int imageheight = int(ceil(inputh/stride));
    // int imagewidth = int(ceil(inputw/stride));    
    return imageheight;
}
int calculateoutputimgwidth(int inputw,int stride){
    // int imageheight = int(ceil(inputh/stride));
    int imagewidth = int(ceil(inputw/stride));    
    return imagewidth;
}

ILayer* pool(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int out_size){
    int inch_h = input.getDimensions().d[1];
    int inch_w = input.getDimensions().d[2];
    int s_h = inch_h / out_size;
    int s_w = inch_w / out_size;
    int k_w = inch_w - (out_size - 1) * s_w;
    int k_h = inch_h - (out_size - 1) * s_h;
    auto pool1 = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{k_w, k_h});
    pool1->setPaddingNd(DimsHW{0, 0});
    pool1->setStrideNd(DimsHW{s_w, s_h});
    return pool1;
}

ILayer* hSwish(INetworkDefinition *network, ITensor& input, std::string name) {
    auto hsig = network->addActivation(input, ActivationType::kHARD_SIGMOID);
    assert(hsig);
    hsig->setAlpha(1.0 / 6.0);
    hsig->setBeta(0.5);
    ILayer* hsw = network->addElementWise(input, *hsig->getOutput(0),ElementWiseOperation::kPROD);
    assert(hsw);
    return hsw;
}


ILayer* DepthWise(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int inp, int oup, int groups,int dwkernelsize) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv3 = addSamePaddingConv(network, weightMap, input,imgh,imgw,dwkernelsize,1,groups,groups,weightMap[lname + "_dwconv/depthwise_kernel:0"],emptywts);
    imgh = calculateoutputimgheight(imgh,1);
    imgw = calculateoutputimgwidth(imgw,1);
    
    
    // network->addConvolutionNd(input, groups, DimsHW{dwkernelsize, dwkernelsize}, weightMap[lname + "_dwconv/depthwise_kernel:0"], emptywts);
    std::cout << "depthwise conv layer" << std::endl;
    assert(conv3);
    // conv3->setStrideNd(DimsHW{1, 1});
    // conv3->setPaddingNd(DimsHW{0, 0});
    // conv3->setNbGroups(groups);
    auto bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "_bn/", 1e-3);
    assert(bn3);
    return bn3;
}





ILayer* stem(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch,std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    Dims diminput = input.getDimensions();
    int windowsizeh = diminput.d[1];
    int windowsizew = diminput.d[2];
    std::cout << "input layer size"<< windowsizeh<<"  "<<windowsizew<< std::endl;
    IConvolutionLayer* conv1 = addSamePaddingConv(network, weightMap, input,imgh,imgw,3,2,outch,1,weightMap[lname + "_conv/kernel:0"],emptywts);
    imgh = calculateoutputimgheight(imgh,2);
    imgw = calculateoutputimgwidth(imgw,2);
    // network->addConvolutionNd(input, outch, DimsHW{3, 3}, weightMap[lname + "_conv/kernel:0"], emptywts);
    // conv1->setStrideNd(DimsHW{2, 2});
    // conv1->setPaddingNd(DimsHW{0, 0});
    std::cout << "conv layer" << std::endl;
    assert(conv1);
    Dims diminput1 = conv1->getOutput(0)->getDimensions();
    int windowsizeh1 = diminput1.d[1];
    int windowsizew1 = diminput1.d[2];
    std::cout << "same conv output layer size"<< windowsizeh1<<"  "<<windowsizew1<< std::endl;
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "_bn/", 1e-5);
    assert(bn1);
    auto hsig = network->addActivation(*bn1->getOutput(0), ActivationType::kHARD_SIGMOID);
    assert(hsig);
    //multiply
    IElementWiseLayer* ew1 = network->addElementWise(*bn1->getOutput(0), *hsig->getOutput(0), ElementWiseOperation::kPROD);
    std::cout << "stem element wise layer" << std::endl;
    return ew1;
}

IActivationLayer* finallayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + "_conv/kernel:0"], emptywts);
    std::cout << "final conv layer" << std::endl;
    assert(conv1);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "_bn/", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    return relu1;
}

ILayer* se_layer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int reduceinch, int reduceoutch, int expandinch, int expandoutch, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    Dims diminput = input.getDimensions();
    int windowsizeh = diminput.d[1];
    int windowsizew = diminput.d[2];
    std::cout << "input layer size"<< windowsizeh<<"  "<<windowsizew<< std::endl;

    // pooling to 1x1
    ILayer* pool1 = network->addPoolingNd(input,PoolingType::kAVERAGE,DimsHW{windowsizeh,windowsizew});
    Dims dim0 = pool1->getOutput(0)->getDimensions();

    std::cout << "pool layer size"<< dim0.d[0]<<" "<<dim0.d[1]<<" "<<dim0.d[2]<< std::endl;
    std::cout << "adaptive pool layer" << std::endl;


    //reduce
    IConvolutionLayer* conv1 = addSamePaddingConv(network, weightMap, input,imgh,imgw,1,1,reduceoutch,1,weightMap[lname + "_reduce/kernel:0"],weightMap[lname + "_reduce/bias:0"]);
    imgh = calculateoutputimgheight(imgh,1);
    imgw = calculateoutputimgwidth(imgw,1);


    // network->addConvolutionNd(*pool1->getOutput(0), reduceoutch, DimsHW{1, 1}, weightMap[lname + "_reduce/kernel:0"], weightMap[lname + "_reduce/bias:0"]);
    std::cout << "reduce conv layer" << std::endl;



    assert(conv1);
    // x*sigmod(x)
    auto hsig = network->addActivation(*conv1->getOutput(0), ActivationType::kHARD_SIGMOID);
    assert(hsig);
    IElementWiseLayer* ew1 = network->addElementWise(*conv1->getOutput(0), *hsig->getOutput(0), ElementWiseOperation::kPROD);

    // expand
    IConvolutionLayer* conv2 = addSamePaddingConv(network, weightMap, *ew1->getOutput(0),imgh,imgw,1,1,expandoutch,1,weightMap[lname + "_expand/kernel:0"],weightMap[lname + "_expand/bias:0"]);
    imgh = calculateoutputimgheight(imgh,1);
    imgw = calculateoutputimgwidth(imgw,1);
    
    // network->addConvolutionNd(*ew1->getOutput(0), expandoutch, DimsHW{1, 1}, weightMap[lname + "_expand/kernel:0"], weightMap[lname + "_expand/bias:0"]);
    std::cout << "expand conv layer" << std::endl;
    assert(conv2);
    return conv2;
}


ILayer* mbconvblocks(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int expand_convinch, int expand_convoutch,  int dwinch, int dwoutch, int se_reduceinch, int se_reduceoutch, int se_expandinch, int se_expandoutch,int project_inch, int project_outch, int ratioequal1,  std::string lname,int dwkernelsize) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    //conv
    IScaleLayer* bn1;
    if (ratioequal1==0){
        IConvolutionLayer* expand = addSamePaddingConv(network, weightMap, input,imgh,imgw,1,1,expand_convoutch,1,weightMap[lname + "_expand_conv/kernel:0"],emptywts);
        imgh = calculateoutputimgheight(imgh,1);
        imgw = calculateoutputimgwidth(imgw,1);
        // network->addConvolutionNd(input, expand_convoutch, DimsHW{1, 1}, weightMap[lname + "_expand_conv/kernel:0"], emptywts);
        std::cout << "first expand conv layer" << std::endl;
        bn1 = addBatchNorm2d(network, weightMap, *expand->getOutput(0), lname + "_expand_bn/", 1e-5);
    }

    ILayer* dw1;
    //depth wise conv
    // dw1->setName("depthwise layer 1");
    if (ratioequal1==0){
        dw1 = DepthWise(network, weightMap, *bn1->getOutput(0), lname, dwinch, dwoutch, dwinch,dwkernelsize);
        // dw1 = DepthWise1(network, weightMap, *bn1->getOutput(0), lname, dwinch, dwoutch, dwinch);
    }else{
        dw1 = DepthWise(network, weightMap, input, lname, dwinch, dwoutch, dwinch,dwkernelsize);
        // dw1 = DepthWise1(network, weightMap, input, lname, dwinch, dwoutch, dwinch);
    }    
    
    assert(dw1);
    ILayer* hsw2;
    auto hsig0 = network->addActivation(*dw1->getOutput(0), ActivationType::kHARD_SIGMOID);
    assert(hsig0);
    
    hsw2 = network->addElementWise(*dw1->getOutput(0), *hsig0->getOutput(0), ElementWiseOperation::kPROD);
    // hsw2->setName("swish add layer 1");
    std::cout << "swish elementwise layer" << std::endl;

    //se layer
    std::cout << "start se layer" << std::endl;
    auto se1 = se_layer(network,weightMap,*hsw2->getOutput(0),se_reduceinch, se_reduceoutch, se_expandinch, se_expandoutch, lname+"_se");
    
    auto hsig = network->addActivation(*se1->getOutput(0), ActivationType::kHARD_SIGMOID);
    // hsig->setReshapeDimensions(Dims3(*hsw2->getDimensions().d[0], *hsw2->getDimensions().d[1], *hsw2->getDimensions().d[2]));
    assert(hsig);

    //multiply
    IElementWiseLayer* ew1 = network->addElementWise(*hsw2->getOutput(0), *hsig->getOutput(0), ElementWiseOperation::kPROD);
    // ew1->setName("multiply layer 1");
    std::cout << "multiply layer" << std::endl;
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *ew1->getOutput(0), lname + "_bn/", 1e-5);

    // project conv
    IConvolutionLayer* conv1 = addSamePaddingConv(network, weightMap, *bn2->getOutput(0),imgh,imgw,1,1,project_outch,1,weightMap[lname + "_project_conv/kernel:0"],emptywts);
    imgh = calculateoutputimgheight(imgh,1);
    imgw = calculateoutputimgwidth(imgw,1);

    Dims diminput1 = conv1->getOutput(0)->getDimensions();
    int windowsizeh1 = diminput1.d[1];
    int windowsizew1 = diminput1.d[2];
    std::cout << "project conv output layer size"<< windowsizeh1<<"  "<<windowsizew1<< std::endl;
    
    // network->addConvolutionNd(*bn2->getOutput(0), project_outch, DimsHW{1, 1}, weightMap[lname + "_project_conv/kernel:0"], emptywts);
    std::cout << "project conv layer" << std::endl;
    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "_project_bn/", 1e-5);
    Dims dim0 = bn3->getOutput(0)->getDimensions();
    std::cout << "project batchnorm layer size"<< dim0.d[0]<<" "<<dim0.d[1]<<" "<<dim0.d[2]<< std::endl;
    //add
    // IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *hsw2->getOutput(0), lname + "bn1", 1e-5);
    // IActivationLayer* relu1 = network->addActivation(*hsw2->getOutput(0), ActivationType::kRELU);
    if (input.getDimensions().d[0] == diminput1.d[0]){
        IElementWiseLayer* ew2 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
        // ew2->setName("final add layer");
        std::cout << "final add layer" << std::endl;
        return ew2;
    }
    else{
        // bn3->setName("not equal, return batchnorm layer");
        std::cout << "batchnorm layer" << std::endl;
        return bn3;
    }

}

ILayer* blocks01(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch,float expandratio,int ratioequal1,int countnum, std::string lname) {
    ILayer* mb1 = mbconvblocks(network,weightMap,input,0,0,40,1,40,10,10,40,40,24,1,lname+"a",3);
    // mb1->setName("block1a layer");
    ILayer* mb2 = mbconvblocks(network,weightMap,*mb1->getOutput(0),0,0,24,1,24,6,6,24,24,24,1,lname+"b",3);
    mb2->setName("block1b layer");
    return mb2;
}

ILayer* blocks02(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch,float expandratio,int ratioequal1,int countnum, std::string lname) {
    ILayer* mb1 = mbconvblocks(network,weightMap,input,24,144,144,1,144,6,6,144,144,32,0,lname+"a",3);
    ILayer* mb2 = mbconvblocks(network,weightMap,*mb1->getOutput(0),32,192,192,1,192,8,8,192,192,32,0,lname+"b",3);
    ILayer* mb3 = mbconvblocks(network,weightMap,*mb2->getOutput(0),32,192,192,1,192,8,8,192,192,32,0,lname+"c",3);
    return mb3;
}

ILayer* blocks03(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch,float expandratio,int ratioequal1,int countnum, std::string lname) {
    ILayer* mb1 = mbconvblocks(network,weightMap,input,32,192,192,1,192,8,8,192,192,48,0,lname+"a",5);
    ILayer* mb2 = mbconvblocks(network,weightMap,*mb1->getOutput(0),48,288,288,1,288,12,12,288,288,48,0,lname+"b",5);
    ILayer* mb3 = mbconvblocks(network,weightMap,*mb2->getOutput(0),48,288,288,1,288,12,12,288,288,48,0,lname+"c",5);
    return mb3;
}

ILayer* blocks04(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch,float expandratio,int ratioequal1,int countnum, std::string lname) {
    ILayer* mb1 = mbconvblocks(network,weightMap,input,48,288,288,1,288,12,12,288,288,96,0,lname+"a",3);
    ILayer* mb2 = mbconvblocks(network,weightMap,*mb1->getOutput(0),96,576,576,1,576,24,24,576,576,96,0,lname+"b",3);
    ILayer* mb3 = mbconvblocks(network,weightMap,*mb2->getOutput(0),96,576,576,1,576,24,24,576,576,96,0,lname+"c",3);
    ILayer* mb4 = mbconvblocks(network,weightMap,*mb3->getOutput(0),96,576,576,1,576,24,24,576,576,96,0,lname+"d",3);
    ILayer* mb5 = mbconvblocks(network,weightMap,*mb4->getOutput(0),96,576,576,1,576,24,24,576,576,96,0,lname+"e",3);
    return mb5;
}

ILayer* blocks05(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch,float expandratio,int ratioequal1,int countnum, std::string lname) {
    ILayer* mb1 = mbconvblocks(network,weightMap,input,96,576,576,1,576,24,24,576,576,136,0,lname+"a",5);
    ILayer* mb2 = mbconvblocks(network,weightMap,*mb1->getOutput(0),136,816,816,1,816,34,34,816,816,136,0,lname+"b",5);
    ILayer* mb3 = mbconvblocks(network,weightMap,*mb2->getOutput(0),136,816,816,1,816,34,34,816,816,136,0,lname+"c",5);
    ILayer* mb4 = mbconvblocks(network,weightMap,*mb3->getOutput(0),136,816,816,1,816,34,34,816,816,136,0,lname+"d",5);
    ILayer* mb5 = mbconvblocks(network,weightMap,*mb4->getOutput(0),136,816,816,1,816,34,34,816,816,136,0,lname+"e",5);
    return mb5;
}

ILayer* blocks06(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch,float expandratio,int ratioequal1,int countnum, std::string lname) {
    ILayer* mb1 = mbconvblocks(network,weightMap,input,136,816,816,1,816,34,34,816,816,232,0,lname+"a",5);
    ILayer* mb2 = mbconvblocks(network,weightMap,*mb1->getOutput(0),232,1392,1392,1,1392,58,58,1392,1392,232,0,lname+"b",5);
    ILayer* mb3 = mbconvblocks(network,weightMap,*mb2->getOutput(0),232,1392,1392,1,1392,58,58,1392,1392,232,0,lname+"c",5);
    ILayer* mb4 = mbconvblocks(network,weightMap,*mb3->getOutput(0),232,1392,1392,1,1392,58,58,1392,1392,232,0,lname+"d",5);
    ILayer* mb5 = mbconvblocks(network,weightMap,*mb4->getOutput(0),232,1392,1392,1,1392,58,58,1392,1392,232,0,lname+"e",5);
    ILayer* mb6 = mbconvblocks(network,weightMap,*mb5->getOutput(0),232,1392,1392,1,1392,58,58,1392,1392,232,0,lname+"f",5);
    return mb6;
}

ILayer* blocks07(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch,float expandratio,int ratioequal1,int countnum, std::string lname) {
    ILayer* mb1 = mbconvblocks(network,weightMap,input,232,1392,1392,1,1392,58,58,1392,1392,384,0,lname+"a",3);
    ILayer* mb2 = mbconvblocks(network,weightMap,*mb1->getOutput(0),384,2304,2304,1,2304,96,96,2304,2304,384,0,lname+"b",3);
    return mb2;
}


// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("/home/pengyuzhou/workspace/tensorrtx_new/tensorrtx/psenet/efficientnetb3.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    std::cout << "start stem " << std::endl;
    ILayer* stem1 = stem(network, weightMap, *data, 3, 40, "stem");
    std::cout << "start blocks " << std::endl;
    auto block1 = blocks01(network, weightMap, *stem1->getOutput(0), 40, 40, 1, 1, 1,"block1");
    auto block2 = blocks02(network, weightMap, *block1->getOutput(0), 64, 64, 1, 0, 2,"block2");
    auto block3 = blocks03(network, weightMap, *block2->getOutput(0), 64, 64, 1, 0, 2,"block3");
    auto block4 = blocks04(network, weightMap, *block3->getOutput(0), 64, 64, 1, 0, 4,"block4");
    auto block5 = blocks05(network, weightMap, *block4->getOutput(0), 64, 64, 1, 0, 4,"block5");
    auto block6 = blocks06(network, weightMap, *block5->getOutput(0), 64, 64, 1, 0, 5,"block6");
    auto block7 = blocks07(network, weightMap, *block6->getOutput(0), 64, 64, 1, 0, 1,"block7");
    std::cout << "start final layer"  << std::endl;
    auto fin = finallayer(network,weightMap,*block7->getOutput(0),384,1536,"top");

    // IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{7, 7}, weightMap["conv1.weight"], emptywts);
    // assert(conv1);
    // conv1->setStrideNd(DimsHW{2, 2});
    // conv1->setPaddingNd(DimsHW{3, 3});

    // IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);

    // IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    // assert(relu1);

    // IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    // assert(pool1);
    // pool1->setStrideNd(DimsHW{2, 2});
    // pool1->setPaddingNd(DimsHW{1, 1});

    // IActivationLayer* relu2 = basicBlock(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "layer1.0.");
    // IActivationLayer* relu3 = basicBlock(network, weightMap, *relu2->getOutput(0), 64, 64, 1, "layer1.1.");

    // IActivationLayer* relu4 = basicBlock(network, weightMap, *relu3->getOutput(0), 64, 128, 2, "layer2.0.");
    // IActivationLayer* relu5 = basicBlock(network, weightMap, *relu4->getOutput(0), 128, 128, 1, "layer2.1.");

    // IActivationLayer* relu6 = basicBlock(network, weightMap, *relu5->getOutput(0), 128, 256, 2, "layer3.0.");
    // IActivationLayer* relu7 = basicBlock(network, weightMap, *relu6->getOutput(0), 256, 256, 1, "layer3.1.");

    // IActivationLayer* relu8 = basicBlock(network, weightMap, *relu7->getOutput(0), 256, 512, 2, "layer4.0.");
    // IActivationLayer* relu9 = basicBlock(network, weightMap, *relu8->getOutput(0), 512, 512, 1, "layer4.1.");

    // IPoolingLayer* pool2 = network->addPoolingNd(*relu9->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    // assert(pool2);
    // pool2->setStrideNd(DimsHW{1, 1});
    
    // IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), 1000, weightMap["fc.weight"], weightMap["fc.bias"]);
    // assert(fc1);

    fin->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*fin->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./efficientnet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./efficientnet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("efficientnet.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("resnet18.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        return -1;
    }


    // Subtract mean from image
    static float data[3 * INPUT_H * INPUT_W];
    for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        data[i] = 1.0;

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Run inference
    static float prob[OUTPUT_SIZE];
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < 10; i++)
    {
        std::cout << prob[i] << ", ";
    }
    std::cout << std::endl;
    for (unsigned int i = 0; i < 10; i++)
    {
        std::cout << prob[OUTPUT_SIZE - 10 + i] << ", ";
    }
    std::cout << std::endl;

    return 0;
}
