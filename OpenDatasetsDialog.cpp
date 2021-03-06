#include "OpenDatasetsDialog.hpp"
#include "StartTrainingDialog.hpp"
#include "ProjectFile.hpp"

#include <opencv2/opencv.hpp>

#include <UNet/TrainUnet2D.hpp>

#include <QtWidgets>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#ifdef _MSC_VER
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include <fstream>
#include <iostream>

namespace bp = boost::property_tree;

namespace std {
template<>
struct less<cv::Vec3b>
{
  bool operator()(::cv::Vec3b const& a, ::cv::Vec3b const& b) const
  {
    return std::tie(a[0], a[1], a[2]) < std::tie(b[0], b[1], b[2]);
  }
};
}

namespace {
auto calculateLabels(std::set<cv::Vec3b>& colorSet, std::string const& labelsFile) -> std::map<cv::Vec3b, std::vector<cv::Rect>>
{
  cv::Mat labelsImage = cv::imread(labelsFile);
  cv::imshow("some", labelsImage);
  cv::waitKey(1);
  for (auto r = 0; r < labelsImage.rows; ++r)
  {
    auto ptr = labelsImage.ptr<cv::Vec3b>(r);
    for(int c = 0; c < labelsImage.cols; c++)
    {
      if (colorSet.count(ptr[c]) == 0)
      {
        colorSet.insert(ptr[c]);
      }
    }
  }
  std::map<cv::Vec3b, std::vector<cv::Rect>> labels;
  for(auto const& color : colorSet)
  {
    cv::Mat currentMask;
    cv::inRange(labelsImage, color, color, currentMask);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(currentMask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    labels[color] = std::vector<cv::Rect>(contours.size());
    for (auto j = 0; j < contours.size(); ++j)
    {
      labels[color][j] = cv::boundingRect(contours[j]);
    }
  }
  return labels;
}

enum { absoluteFileNameRole = Qt::UserRole + 1 };

QString fileNameOfItem(const QTableWidgetItem *item)
{
  return item->data(absoluteFileNameRole).toString();
}

void openFile(const QString &fileName)
{
  QDesktopServices::openUrl(QUrl::fromLocalFile(fileName));
}
} /// end namespace anonymous

OpenDatasetsDialog::OpenDatasetsDialog(std::string const& projectFile, QWidget* parent)
   : QDialog(parent)
{
   setWindowTitle(tr("Open datasets dialog"));

   _unet = std::make_unique<UNet>("/home/oleksandr/WORK/09_05_2021/unet_training_tool/unet_3c1cl3l8f.cfg",
                                  "/home/oleksandr/WORK/09_05_2021/unet_training_tool/checkpoints_3c1cl3l8f/best_28.weights",
                                  cv::Size{8, 8},
                                  std::vector<float>{0.99},
                                  true);
   _projectFile = projectFile;

   labelsTable = new QTableWidget(0, 3);
   labelsTable->setSelectionBehavior(QAbstractItemView::SelectRows);

   QStringList labels;
   labels << tr("Added") << tr("Filename") << tr("Size");
   labelsTable->setHorizontalHeaderLabels(labels);
   labelsTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
   labelsTable->verticalHeader()->hide();
   labelsTable->setShowGrid(false);
   labelsTable->setContextMenuPolicy(Qt::CustomContextMenu);
   connect(labelsTable, &QTableWidget::currentCellChanged, this, &OpenDatasetsDialog::openDatasetItem);

   _labelsViewLabel = new QLabel(this);
   _scrollArea = new QScrollArea(this);

   _labelsViewLabel->setBackgroundRole(QPalette::Base);
   _labelsViewLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
   _labelsViewLabel->setScaledContents(true);

   _scrollArea->setBackgroundRole(QPalette::Dark);
   _scrollArea->setWidget(_labelsViewLabel);
   _scrollArea->setVisible(true);

   classCountTable = new QTableWidget(0, 4);
   QStringList classCountTableLabels;
   classCountTableLabels << tr("Added") << tr("Class color") << tr("Class name") << tr("Count");
   classCountTable->setHorizontalHeaderLabels(classCountTableLabels);
   classCountTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
   classCountTable->verticalHeader()->hide();
   classCountTable->setShowGrid(false);

   connect(classCountTable, &QTableWidget::cellDoubleClicked, [this](int row, int column) {
     switch(column)
     {
       case 0:
         break;
       case 1:
         {
           auto color = QColorDialog::getColor(Qt::white, nullptr, tr("Select label color"));
           if(color.isValid())
           {
             classCountTable->item(row, 1)->setBackgroundColor(color);
             updateColorMaps();
           }
         }
         break;
       case 2:
         break;
       case 3:
       default:
         break;
     }
   });

   _startTrainingButton = new QPushButton(tr("&Start training..."), this);
   connect(_startTrainingButton, &QAbstractButton::clicked, [this](){
     auto startTrainingDialog = new StartTrainingDialog(_projectFile, this);
     startTrainingDialog->exec();
   });

   auto mainLayout = new QGridLayout(this);
   mainLayout->addWidget(labelsTable, 0, 0);
   mainLayout->addWidget(classCountTable, 1, 0);
   mainLayout->addWidget(_startTrainingButton, 2, 0);
   mainLayout->addWidget(_scrollArea, 0, 1, 3, 1);

   connect(new QShortcut(QKeySequence::Quit, this), &QShortcut::activated, qApp, &QApplication::quit);

   resize(QGuiApplication::primaryScreen()->availableSize() * 3 / 5);
   openViewer(projectFile);
   boost::property_tree::read_json(_projectFile, _pt);
   _classesToColorsMap = ProjectFile::loadColors(_pt);
   for (int i = 0; i < classCountTable->rowCount(); ++i)
   {
     cv::Scalar color = _classesToColorsMap[classCountTable->item(i, 2)->text().toStdString()];
     classCountTable->item(i, 1)->setBackground(QColor(color[2], color[1], color[0]));
   }
   //updateColorMaps();
}

void OpenDatasetsDialog::updateColorMaps()
{
  _classesToColorsMap.clear();
  for (int i = 0; i < classCountTable->rowCount(); ++i)
  {
    auto className = classCountTable->item(i, 2)->text().toStdString();
    auto color = classCountTable->item(i, 1)->backgroundColor();
    _classesToColorsMap[className] = cv::Scalar(color.blue(), color.green(), color.red());
  }
  ProjectFile::saveColors(_pt, _classesToColorsMap);
  bp::write_json(_projectFile, _pt);
  if ((labelsTable->currentRow() >= 0) && (labelsTable->currentRow() < labelsTable->rowCount()))
  {
    openDatasetItem(labelsTable->currentRow(), 0, 0, 0);
  }
}

void OpenDatasetsDialog::openViewer(std::string const& projectFile)
{
  labelsTable->setRowCount(0);
  classCountTable->setRowCount(0);

  std::map<cv::Vec3b, std::vector<cv::Rect>> allLabels;
  std::map<std::string, uint32_t> allLabelsByName;
  std::set<cv::Vec3b> colorSet;

  bp::read_json(projectFile, _pt);
  ProjectFile::iterateOverDatasets(_pt, [&](std::string const& imagesDirercoryPath, std::string const& labelsDirectoryPath) {
    openCurrentDataset(imagesDirercoryPath, labelsDirectoryPath, allLabels, allLabelsByName, colorSet);
  });

  for (auto const& classLabels : allLabels)
  {
    auto classNameItem = new QTableWidgetItem(QString::fromStdString(std::to_string(classLabels.first[0]) + " " +
                                                                                  std::to_string(classLabels.first[1]) + " " +
                                                                                  std::to_string(classLabels.first[2])));
    classNameItem->setFlags(classNameItem->flags() ^ Qt::ItemIsEditable);
    classNameItem->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);

    auto countItem = new QTableWidgetItem(QString::fromStdString(std::to_string(classLabels.second.size())));
    countItem->setFlags(countItem->flags() ^ Qt::ItemIsEditable);
    countItem->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);

    auto addedItem = new QTableWidgetItem(tr("Added"));
    addedItem->setFlags(addedItem->flags() | Qt::ItemIsUserCheckable);
    addedItem->setCheckState(Qt::Checked);

    auto classColorItem = new QTableWidgetItem(tr(""));
    classColorItem->setBackground(QColor(classLabels.first[0], classLabels.first[1], classLabels.first[2]));

    int row = classCountTable->rowCount();
    classCountTable->insertRow(row);
    classCountTable->setItem(row, 0, addedItem);
    classCountTable->setItem(row, 1, classColorItem);
    classCountTable->setItem(row, 2, classNameItem);
    classCountTable->setItem(row, 3, countItem);
  }

  for (auto const& classLabels : allLabelsByName)
  {
    auto classNameItem = new QTableWidgetItem(QString::fromStdString(classLabels.first));
    classNameItem->setFlags(classNameItem->flags() ^ Qt::ItemIsEditable);
    classNameItem->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);

    auto countItem = new QTableWidgetItem(QString::fromStdString(std::to_string(classLabels.second)));
    countItem->setFlags(countItem->flags() ^ Qt::ItemIsEditable);
    countItem->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);

    auto addedItem = new QTableWidgetItem(tr("Added"));
    addedItem->setFlags(addedItem->flags() | Qt::ItemIsUserCheckable);
    addedItem->setCheckState(Qt::Checked);

    auto classColorItem = new QTableWidgetItem(tr(""));
    classColorItem->setBackground(QColor(QColor::colorNames().first()));

    int row = classCountTable->rowCount();
    classCountTable->insertRow(row);
    classCountTable->setItem(row, 0, addedItem);
    classCountTable->setItem(row, 1, classColorItem);
    classCountTable->setItem(row, 2, classNameItem);
    classCountTable->setItem(row, 3, countItem);
  }
  //updateColorMaps();
}

void OpenDatasetsDialog::openCurrentDataset(std::string const& imagesDirectoryPath,
                                            std::string const& labelsDirectoryPath,
                                            std::map<cv::Vec3b, std::vector<cv::Rect>>& allLabels,
                                            std::map<std::string, uint32_t>& allLabelsByName,
                                            std::set<cv::Vec3b>& colorSet)
{
   auto imagesDirectoryPathCopy = imagesDirectoryPath;
   std::replace(imagesDirectoryPathCopy.begin(), imagesDirectoryPathCopy.end(), '\\', '/');
   auto lastSlash = imagesDirectoryPathCopy.find_last_of("/");
   auto imagesDirectoryPathFixed = fs::path(imagesDirectoryPath).is_absolute() ? imagesDirectoryPath : labelsDirectoryPath + "/" + imagesDirectoryPathCopy.substr(0, lastSlash);
   std::vector<fs::directory_entry> imageList(fs::directory_iterator{imagesDirectoryPathFixed}, fs::directory_iterator{});

   QProgressDialog progressDialog(this);
   progressDialog.setCancelButtonText(tr("&Cancel"));
   progressDialog.setRange(0, imageList.size());
   progressDialog.setWindowTitle(tr("Counting labels"));

   auto currentLabel = 0;
   bool splitDataset = true;
   QString dir = QFileDialog::getExistingDirectory(this, tr("Open directory for saving split dataset"),
                                                  ".",
                                                  QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
   if (dir.isEmpty())
   {
     QMessageBox msgBox;
     msgBox.setText("Folder for saving split dataset was not selected, this operation will be skipped!");
     msgBox.exec();
     splitDataset = false;
     //return;
   }
   for (auto const& file : imageList)
   {
     auto filename = file.path().filename().string();
     auto fileext = file.path().extension().string();
     filename = filename.substr(0,filename.find_last_of('.'));
     auto testFolderPath = labelsDirectoryPath + "/" + filename + ".json";
     std::cout << testFolderPath << std::endl;
     auto res = fs::exists(testFolderPath);
     if (fs::exists(labelsDirectoryPath + "/" + filename + fileext))
     {
       _dataset.emplace_back(file.path().string(), labelsDirectoryPath + "/" + filename + fileext);
       auto currentLabels = calculateLabels(colorSet, _dataset.back().second);
       for (auto const& rects : currentLabels)
       {
         allLabels[rects.first].insert(allLabels[rects.first].end(), rects.second.cbegin(), rects.second.cend());
       }
     }
     else if (fs::exists(labelsDirectoryPath + "/" + filename + ".json"))
     {
       _dataset.emplace_back(file.path().string(), labelsDirectoryPath + "/" + filename + ".json");
       LabelMeDeleteImage(_dataset.back().second);
       ////
       std::string const splitDir = dir.toStdString();
       std::map<std::string, uint32_t> classesExist;
       if (!CountingLabeledObjects(classesExist, _dataset.back().second, true))
       {
         QMessageBox msgBox;
         msgBox.setText(QString::fromStdString(std::string("The file: ") + labelsDirectoryPath + "/" + filename + ".json" + "is wrong!"));
         msgBox.exec();
         continue;
       }
       for (auto const& classExist : classesExist)
       {
         if (splitDataset)
         {
             fs::create_directories(splitDir + "/" + classExist.first + "/images");
             fs::create_directories(splitDir + "/" + classExist.first + "/data");
         }
         if (classesExist.find("trash") != classesExist.end())
         {
           if (splitDataset)
           {
               fs::copy(_dataset.back().first, splitDir + "/" + "trash" + "/images");
               fs::copy(_dataset.back().second, splitDir + "/" + "trash" + "/data");
               bp::ptree ptData;
               auto const newImagePath = splitDir + "/" + "trash" + "/images/" + fs::path(_dataset.back().first).filename().string();
               auto const newDataPath = splitDir + "/" + "trash" + "/data/" + fs::path(_dataset.back().second).filename().string();
               bp::read_json(newDataPath, ptData);
               ptData.put<std::string>("imagePath", newImagePath);
               bp::write_json(newDataPath, ptData);
           }
           break;
         }
         if (splitDataset)
         {
             fs::copy(_dataset.back().first, splitDir + "/" + classExist.first + "/images");
             fs::copy(_dataset.back().second, splitDir + "/" + classExist.first + "/data");
             bp::ptree ptData;
             auto const newImagePath = splitDir + "/" + classExist.first + "/images/" + fs::path(_dataset.back().first).filename().string();
             auto const newDataPath = splitDir + "/" + classExist.first + "/data/" + fs::path(_dataset.back().second).filename().string();
             bp::read_json(newDataPath, ptData);
             ptData.put<std::string>("imagePath", newImagePath);
             bp::write_json(newDataPath, ptData);
         }
       }
       if (!CountingLabeledObjects(allLabelsByName, _dataset.back().second, true))
       {
         QMessageBox msgBox;
         msgBox.setText(QString::fromStdString(std::string("The file: ") + labelsDirectoryPath + "/" + filename + ".json" + "is wrong!"));
         msgBox.exec();
         continue;
       }
     }
     else
     {
         continue;
     }
     auto filePathQ = QString::fromStdString(_dataset.back().first);
     const QString toolTip = QDir::toNativeSeparators(filePathQ);
     const QString relativePath = QDir::toNativeSeparators(currentDir.relativeFilePath(filePathQ));
     const qint64 size = QFileInfo(filePathQ).size();
     auto addedItem = new QTableWidgetItem(tr("Added"));
     addedItem->setFlags(addedItem->flags() | Qt::ItemIsUserCheckable);
     addedItem->setCheckState(Qt::Checked);
     auto fileNameItem = new QTableWidgetItem(relativePath);
     fileNameItem->setData(absoluteFileNameRole, QVariant(filePathQ));
     fileNameItem->setToolTip(toolTip);
     fileNameItem->setFlags(fileNameItem->flags() ^ Qt::ItemIsEditable);
     auto sizeItem = new QTableWidgetItem(QString::fromStdString(std::to_string(size)));
     sizeItem->setData(absoluteFileNameRole, QVariant(filePathQ));
     sizeItem->setToolTip(toolTip);
     sizeItem->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
     sizeItem->setFlags(sizeItem->flags() ^ Qt::ItemIsEditable);
     int row = labelsTable->rowCount();
     labelsTable->insertRow(row);
     labelsTable->setItem(row, 0, addedItem);
     labelsTable->setItem(row, 1, fileNameItem);
     labelsTable->setItem(row, 2, sizeItem);

     progressDialog.setValue(currentLabel);
     progressDialog.setLabelText(tr("Processed label number %1 of %n...", nullptr, imageList.size()).arg(currentLabel++));
     QCoreApplication::processEvents();

     if (progressDialog.wasCanceled())
     {
       break;
     }
   }
}

void OpenDatasetsDialog::openDatasetItem(int row, int, int, int)
{
  cv::Mat frame = cv::imread(_dataset[row].first, cv::IMREAD_COLOR);
  auto extention = _dataset[row].second.substr(_dataset[row].second.find_last_of('.') + 1);
  cv::Mat labelsImage = (extention == "json") ? ConvertPolygonsToMask(_dataset[row].second, _classesToColorsMap) : cv::imread(_dataset[row].second);
  if (labelsImage.empty())
  {
    QMessageBox msgBox;
    msgBox.setText(QString::fromStdString(std::string("Something wrong with annotation: ") + _dataset[row].second));
    msgBox.exec();
  }
  else
  {
    cv::addWeighted(frame, 1.0, labelsImage, 0.5, 0.0, frame);
  }
  std::vector<cv::Mat> predictedImages = _unet ? _unet->performPrediction(frame, [](std::vector<cv::Mat> const&){}, true, false) : std::vector<cv::Mat>{};
  for (auto const& predictedImage : predictedImages) {
      cv::imshow("test", predictedImage);
  }
  cv::waitKey(1);
  cv::Rect unionBox;
  auto boundingBoxesAll = UNet::foundBoundingBoxes(predictedImages);
  for (auto& boundingBoxesClass : boundingBoxesAll) {
      std::sort(boundingBoxesClass.begin(), boundingBoxesClass.end(), [](auto& a, auto& b){
          return a.area() > b.area();
      });
      for(auto const& boundingBox : boundingBoxesClass) {
          unionBox |= boundingBox;
          break;
      }
  }
  cv::rectangle(frame, unionBox, cv::Scalar(255, 255, 255), 4);
  image = QImage((uchar*)frame.data, frame.cols, frame.rows, frame.step, QImage::Format_BGR888);
  _labelsViewLabel->setPixmap(QPixmap::fromImage(image)/*.scaled(_labelsViewLabel->width(), _labelsViewLabel->height(), Qt::KeepAspectRatio)*/);
  _labelsViewLabel->adjustSize();
}

void OpenDatasetsDialog::createDatasetLists()
{
   auto imgsList = std::ofstream("imgs.txt");
   auto masksList = std::ofstream("masks.txt");
   auto ignoredImgsList = std::ofstream("ignoredImgs.txt");
   auto ignoredMasksList = std::ofstream("ignoredMasks.txt");
   int rowCount = labelsTable->rowCount();
   for (auto i = 0; i < rowCount; ++i)
   {
      QTableWidgetItem* pItem(labelsTable->item(i, 0));
      if (pItem)
      {
         Qt::CheckState st = pItem->checkState();
         if (st == Qt::CheckState::Checked)
         {
            imgsList << _dataset[i].first << std::endl;
            masksList << _dataset[i].second << std::endl;
         }
         else
         {
            ignoredImgsList << _dataset[i].first << std::endl;
            ignoredMasksList << _dataset[i].second << std::endl;
         }
      }
   }
}