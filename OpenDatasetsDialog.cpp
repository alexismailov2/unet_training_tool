#include "OpenDatasetsDialog.hpp"
#include "StartTrainingDialog.hpp"

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

void iterateOverDatasets(bp::ptree& pt, std::function<void(std::string const&, std::string const&)>&& cb)
{
  auto datasets = pt.get_child_optional("datasets");
  if (datasets.has_value())
  {
    for (auto const& item : datasets.get())
    {
      auto images = item.second.get_optional<std::string>("images");
      auto annotations = item.second.get_optional<std::string>("annotations");
      if (images.has_value() && annotations.has_value())
      {
        cb(images.get(), annotations.get());
      }
    }
  }
}

void saveColors(bp::ptree& tree, std::map<std::string, cv::Scalar> const& colorMap)
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

auto loadColors(bp::ptree& tree) -> std::map<std::string, cv::Scalar>
{
  std::map<std::string, cv::Scalar> classesColors;
  auto datasets = tree.get_child_optional("classesColorsMap");
  if (datasets.has_value())
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
} /// end namespace anonymous

OpenDatasetsDialog::OpenDatasetsDialog(std::string const& projectFile, QWidget* parent)
   : QDialog(parent)
{
   setWindowTitle(tr("Open datasets dialog"));

   _projectFile = projectFile;

   createDatasetButton = new QPushButton(tr("&Create dataset lists"), this);
   connect(createDatasetButton, &QAbstractButton::clicked, this, &OpenDatasetsDialog::createDatasetLists);

//   imagesDirectoryComboBox = createComboBox(QDir::toNativeSeparators(QDir::currentPath()));
//   connect(imagesDirectoryComboBox->lineEdit(), &QLineEdit::returnPressed, [this](){
//     openViewerButton->animateClick();
//   });
//   labelsDirectoryComboBox = createComboBox(QDir::toNativeSeparators(QDir::currentPath()));
//   connect(labelsDirectoryComboBox->lineEdit(), &QLineEdit::returnPressed, [this](){
//     openViewerButton->animateClick();
//   });

 //  framesCutLabel = new QLabel("", this);

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
     auto startTrainignDialog = new StartTrainingDialog(_projectFile, this);
     startTrainignDialog->exec();
   });

   QGridLayout* mainLayout = new QGridLayout(this);
   mainLayout->addWidget(labelsTable, 0, 0, 1, 3);
   mainLayout->addWidget(classCountTable, 1, 0, 1, 3);
   //mainLayout->addWidget(framesCutLabel, 2, 0, 1, 3);
   //mainLayout->addWidget(openViewerButton, 3, 2);
   mainLayout->addWidget(createDatasetButton, 2, 1);
   mainLayout->addWidget(_startTrainingButton, 3, 1);
   mainLayout->addWidget(_scrollArea, 0, 4, 3, 1);

   connect(new QShortcut(QKeySequence::Quit, this), &QShortcut::activated, qApp, &QApplication::quit);

   resize(QGuiApplication::primaryScreen()->availableSize() * 3 / 5);
   openViewer(projectFile);
   boost::property_tree::read_json(_projectFile, _pt);
   _classesToColorsMap = loadColors(_pt);
   updateColorMaps();
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
  saveColors(_pt, _classesToColorsMap);
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
  iterateOverDatasets(_pt, [&](std::string const& imagesDirercoryPath, std::string const& labelsDirectoryPath) {
    openCurrentDataset(imagesDirercoryPath, labelsDirectoryPath, allLabels, allLabelsByName, colorSet);
  });

  for (auto const& classLabels : allLabels)
  {
    QTableWidgetItem* classNameItem = new QTableWidgetItem(QString::fromStdString(std::to_string(classLabels.first[0]) + " " +
                                                                                  std::to_string(classLabels.first[1]) + " " +
                                                                                  std::to_string(classLabels.first[2])));
    classNameItem->setFlags(classNameItem->flags() ^ Qt::ItemIsEditable);
    classNameItem->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);

    QTableWidgetItem* countItem = new QTableWidgetItem(QString::fromStdString(std::to_string(classLabels.second.size())));
    countItem->setFlags(countItem->flags() ^ Qt::ItemIsEditable);
    countItem->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);

    QTableWidgetItem* addedItem = new QTableWidgetItem(tr("Added"));
    addedItem->setFlags(addedItem->flags() | Qt::ItemIsUserCheckable);
    addedItem->setCheckState(Qt::Checked);

    QTableWidgetItem* classColorItem = new QTableWidgetItem(tr(""));
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
    QTableWidgetItem* classNameItem = new QTableWidgetItem(QString::fromStdString(classLabels.first));
    classNameItem->setFlags(classNameItem->flags() ^ Qt::ItemIsEditable);
    classNameItem->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);

    QTableWidgetItem* countItem = new QTableWidgetItem(QString::fromStdString(std::to_string(classLabels.second)));
    countItem->setFlags(countItem->flags() ^ Qt::ItemIsEditable);
    countItem->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);

    QTableWidgetItem* addedItem = new QTableWidgetItem(tr("Added"));
    addedItem->setFlags(addedItem->flags() | Qt::ItemIsUserCheckable);
    addedItem->setCheckState(Qt::Checked);

    QTableWidgetItem* classColorItem = new QTableWidgetItem(tr(""));
    classColorItem->setBackground(QColor(QColor::colorNames().first()));

    int row = classCountTable->rowCount();
    classCountTable->insertRow(row);
    classCountTable->setItem(row, 0, addedItem);
    classCountTable->setItem(row, 1, classColorItem);
    classCountTable->setItem(row, 2, classNameItem);
    classCountTable->setItem(row, 3, countItem);
  }
  updateColorMaps();
}

void OpenDatasetsDialog::openCurrentDataset(std::string const& imagesDirectoryPath,
                                            std::string const& labelsDirectoryPath,
                                            std::map<cv::Vec3b, std::vector<cv::Rect>>& allLabels,
                                            std::map<std::string, uint32_t>& allLabelsByName,
                                            std::set<cv::Vec3b>& colorSet)
{
   std::vector<fs::directory_entry> imageList(fs::directory_iterator{imagesDirectoryPath}, fs::directory_iterator{});

   QProgressDialog progressDialog(this);
   progressDialog.setCancelButtonText(tr("&Cancel"));
   progressDialog.setRange(0, imageList.size());
   progressDialog.setWindowTitle(tr("Counting labels"));

   auto currentLabel = 0;
   for (auto file : imageList)
   {
     auto filename = file.path().filename().string();
     filename = filename.substr(0,filename.find_last_of('.'));
     if (fs::exists(labelsDirectoryPath + "/" + filename + ".png"))
     {
       _dataset.emplace_back(file.path().string(), labelsDirectoryPath + "/" + filename + ".png");
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
       if (!CountingLabeledObjects(allLabelsByName, _dataset.back().second))
       {
         QMessageBox msgBox;
         msgBox.setText(QString::fromStdString(std::string("The file: ") + labelsDirectoryPath + "/" + filename + ".json" + "is wrong!"));
         msgBox.exec();
       }
     }
     auto filePathQ = QString::fromStdString(_dataset.back().first);
     const QString toolTip = QDir::toNativeSeparators(filePathQ);
     const QString relativePath = QDir::toNativeSeparators(currentDir.relativeFilePath(filePathQ));
     const qint64 size = QFileInfo(filePathQ).size();
     QTableWidgetItem* addedItem(new QTableWidgetItem(tr("Added")));
     addedItem->setFlags(addedItem->flags() | Qt::ItemIsUserCheckable);
     addedItem->setCheckState(Qt::Checked);
     QTableWidgetItem *fileNameItem = new QTableWidgetItem(relativePath);
     fileNameItem->setData(absoluteFileNameRole, QVariant(filePathQ));
     fileNameItem->setToolTip(toolTip);
     fileNameItem->setFlags(fileNameItem->flags() ^ Qt::ItemIsEditable);
     QTableWidgetItem *sizeItem = new QTableWidgetItem(QString::fromStdString(std::to_string(size)));
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

auto OpenDatasetsDialog::createComboBox(const QString &text) -> QComboBox*
{
   QComboBox *comboBox = new QComboBox;
   comboBox->setEditable(true);
   comboBox->addItem(text);
   comboBox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
   return comboBox;
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
  image = QImage((uchar*)frame.data, frame.cols, frame.rows, frame.step, QImage::Format_BGR888);
  _labelsViewLabel->setPixmap(QPixmap::fromImage(image));
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