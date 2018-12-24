/*************************************************************************
    > File Name: guidedfilter.h
    > Author: 子弹给自由
    > Mail: 374390757@qq.com 
    > Created Time: 2018年12月13日 星期四 17时06分49秒
 ************************************************************************/

#ifndef GUIDED_FILTER_H
#define GUIDED_FILTER_H

#include <opencv2/opencv.hpp>

class GuidedFilterImpl;

class GuidedFilter
{
public:
    GuidedFilter(const cv::Mat &I, int r, double eps);
    ~GuidedFilter();

    cv::Mat filter(const cv::Mat &p, int depth = -1) const;

private:
    GuidedFilterImpl *impl_;
};

cv::Mat guidedFilter(const cv::Mat &I, const cv::Mat &p, int r, double eps, int depth = -1);

#endif