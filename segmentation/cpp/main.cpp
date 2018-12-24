/*************************************************************************
    > File Name: main.cpp
    > Author: 子弹给自由
    > Mail: 374390757@qq.com 
    > Created Time: 2018年12月13日 星期四 17时02分35秒
 ************************************************************************/

#include<iostream>
using namespace std;

#include "globalmatting.h"

// you can get the guided filter implementation
// from https://github.com/atilimcetin/guided-filter
#include "guidedfilter.h"

int main()
{
    cv::Mat image = cv::imread("GT04-image.png", CV_LOAD_IMAGE_COLOR);
    cv::Mat trimap = cv::imread("GT04-trimap.png", CV_LOAD_IMAGE_GRAYSCALE);

    // (optional) exploit the affinity of neighboring pixels to reduce the 
    // size of the unknown region. please refer to the paper
    // 'Shared Sampling for Real-Time Alpha Matting'.
    expansionOfKnownRegions(image, trimap, 9);

    cv::Mat foreground, alpha;
    globalMatting(image, trimap, foreground, alpha);

    // filter the result with fast guided filter
    alpha = guidedFilter(image, alpha, 10, 1e-5);
    for (int x = 0; x < trimap.cols; ++x)
        for (int y = 0; y < trimap.rows; ++y)
        {
            if (trimap.at<uchar>(y, x) == 0)
                alpha.at<uchar>(y, x) = 0;
            else if (trimap.at<uchar>(y, x) == 255)
                alpha.at<uchar>(y, x) = 255;
        }

    cv::imwrite("GT04-alpha.png", alpha);

    return 0;
}