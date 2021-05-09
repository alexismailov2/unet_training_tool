#pragma once

#include <QDialog>
#include <QDir>

#include <opencv_unet/UNet.hpp>

#include <opencv2/core/types.hpp>

#include <boost/property_tree/ptree.hpp>

#include <set>


QT_BEGIN_NAMESPACE
class QComboBox;
class QLabel;
class QPushButton;
class QTableWidget;
class QTableWidgetItem;
class QScrollArea;
QT_END_NAMESPACE

class OpenDatasetsDialog : public QDialog
{
Q_OBJECT

public:
    OpenDatasetsDialog(std::string const& projectFile, QWidget *parent = nullptr);

private slots:
    void createDatasetLists();
    void openDatasetItem(int row, int, int, int);

private:
    void updateColorMaps();
    void openViewer(std::string const& projectFile);
    // TODO: dirty function should be rewritten more clear
    void openCurrentDataset(std::string const& imagesDirectoryPath,
                            std::string const& labelsDirectoryPath,
                            std::map<cv::Vec3b, std::vector<cv::Rect>>& allLabels,
                            std::map<std::string, uint32_t>& allLabelsByName,
                            std::set<cv::Vec3b>& colorSet);

    //QPushButton* _createDatasetButton{};
    QTableWidget* labelsTable{};
    QTableWidget* classCountTable{};
    QDir currentDir;

    QLabel* _labelsViewLabel{};
    QScrollArea* _scrollArea{};
    QImage image;

    std::map<std::string, cv::Scalar> _classesToColorsMap;
    std::vector<std::pair<std::string, std::string>> _dataset;

    std::string _projectFile;
    boost::property_tree::ptree _pt;

    QPushButton* _startTrainingButton{};
    UNet _unet;
};
