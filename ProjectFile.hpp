#pragma once

#include <opencv2/opencv.hpp>

#include <boost/property_tree/ptree.hpp>

#include <map>
#include <string>

namespace bp = boost::property_tree;

struct ProjectFile
{
static auto loadColors(bp::ptree& tree) -> std::map<std::string, cv::Scalar>;
static void saveColors(bp::ptree& tree, std::map<std::string, cv::Scalar> const& colorMap);
static void iterateOverDatasets(bp::ptree& pt, std::function<void(std::string const&, std::string const&)>&& cb);
};