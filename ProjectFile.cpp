#include "ProjectFile.hpp"

#include <boost/property_tree/ptree.hpp>
#include <optional>
#include <map>
#include <string>

auto ProjectFile::loadColors(boost::property_tree::ptree& tree) -> std::map<std::string, cv::Scalar>
{
  std::map<std::string, cv::Scalar> classesColors;
  auto datasets = tree.get_child_optional("classesColorsMap");
  if (datasets.is_initialized())
  {
    for (auto const& item : datasets.get())
    {
      classesColors[item.second.get<std::string>("className")] = cv::Scalar(
        item.second.get<uint8_t>("blue"),
        item.second.get<uint8_t>("green"),
        item.second.get<uint8_t>("red")
      );
    }
  }
  return classesColors;
}

void ProjectFile::saveColors(boost::property_tree::ptree& tree, const std::map<std::string, cv::Scalar>& colorMap)
{
  tree.put_child("classesColorsMap", bp::ptree{});
  auto datasets = tree.get_child_optional("classesColorsMap");
  for (auto const& item : colorMap)
  {
    bp::ptree classColorItem;
    classColorItem.put("className", item.first);
    classColorItem.put("blue", item.second[0]);
    classColorItem.put("green", item.second[1]);
    classColorItem.put("red", item.second[2]);
    datasets.get().push_back(bp::ptree::value_type("", classColorItem));
  }
}

void ProjectFile::iterateOverDatasets(boost::property_tree::ptree& pt,
                                      std::function<void(const std::string&, const std::string&)>&& cb)
{
  auto datasets = pt.get_child_optional("datasets");
  if (datasets.is_initialized())
  {
    for (auto const& item : datasets.get())
    {
      auto images = item.second.get_optional<std::string>("images");
      auto annotations = item.second.get_optional<std::string>("annotations");
      if (images.is_initialized() && annotations.is_initialized())
      {
        std::string imagesPath = images.get();
        std::replace(imagesPath.begin(), imagesPath.end(), '\\', '/');
        std::string annotationsPath = annotations.get();
        std::replace(annotationsPath.begin(), annotationsPath.end(), '\\', '/');
        cb(imagesPath, annotationsPath);
      }
    }
  }
}
