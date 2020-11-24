#pragma once

#include <QDialog>
#include <QDir>

#include <opencv2/core/types.hpp>
#include <boost/property_tree/ptree.hpp>

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
    auto createComboBox(const QString &text = QString()) -> QComboBox*;
    void updateColorMaps();
    void openViewer(std::string const& projectFile);
    void openCurrentDataset(std::string const& imagesDirectoryPath,
                            std::string const& labelsDirectoryPath,
                            std::map<cv::Vec3b, std::vector<cv::Rect>>& allLabels,
                            std::map<std::string, uint32_t>& allLabelsByName,
                            std::set<cv::Vec3b>& colorSet);

    //QComboBox* imagesDirectoryComboBox{};
    //QComboBox* labelsDirectoryComboBox{};
    //QLabel* framesCutLabel{};
    //QPushButton *openViewerButton{};
    QPushButton* createDatasetButton{};
    QTableWidget* labelsTable{};
    QTableWidget* classCountTable{};
    std::vector<std::pair<std::string, std::string>> _dataset;
    QDir currentDir;

    QLabel* _labelsViewLabel{};
    QScrollArea* _scrollArea{};
    QImage image;

    std::map<std::string, cv::Scalar> _classesToColorsMap;

    std::string _projectFile;
    boost::property_tree::ptree _pt;

    QPushButton* _startTrainingButton{};
};
