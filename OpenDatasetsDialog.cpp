#include <QtWidgets>
#include <iostream>

#include "OpenDatasetsDialog.hpp"

#include <opencv2/opencv.hpp>

#ifdef _MSC_VER
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

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
} /// end namespace anonymous

enum { absoluteFileNameRole = Qt::UserRole + 1 };

static inline QString fileNameOfItem(const QTableWidgetItem *item)
{
   return item->data(absoluteFileNameRole).toString();
}

static inline void openFile(const QString &fileName)
{
   QDesktopServices::openUrl(QUrl::fromLocalFile(fileName));
}

OpenDatasetsDialog::OpenDatasetsDialog(QWidget *parent)
   : QDialog(parent)
{
   setWindowTitle(tr("Open datasests dialog"));
   QPushButton *imagesDirectoryButton = new QPushButton(tr("&Browse..."), this);
   connect(imagesDirectoryButton, &QAbstractButton::clicked, [this](){
     QString imagesDirectory = QDir::toNativeSeparators(QFileDialog::getExistingDirectory(this, tr("Open images directory"), QDir::currentPath()));
     if (!imagesDirectory.isEmpty())
     {
        imagesDirectoryComboBox->addItem(imagesDirectory);
        imagesDirectoryComboBox->setCurrentIndex(imagesDirectoryComboBox->findText(imagesDirectory));
     }
   });
   QPushButton *labelsDirectoryButton = new QPushButton(tr("&Browse..."), this);
   connect(labelsDirectoryButton, &QAbstractButton::clicked, [this](){
     QString labelsDirectory = QDir::toNativeSeparators(QFileDialog::getExistingDirectory(this, tr("Open labels directory"), QDir::currentPath()));
     if (!labelsDirectory.isEmpty())
     {
       labelsDirectoryComboBox->addItem(labelsDirectory);
       labelsDirectoryComboBox->setCurrentIndex(labelsDirectoryComboBox->findText(labelsDirectory));
     }
   });
   openViewerButton = new QPushButton(tr("&Open viewer"), this);
   connect(openViewerButton, &QAbstractButton::clicked, this, &OpenDatasetsDialog::openViewer);

   imagesDirectoryComboBox = createComboBox(QDir::toNativeSeparators(QDir::currentPath()));
   connect(imagesDirectoryComboBox->lineEdit(), &QLineEdit::returnPressed, [this](){
     openViewerButton->animateClick();
   });
   labelsDirectoryComboBox = createComboBox(QDir::toNativeSeparators(QDir::currentPath()));
   connect(labelsDirectoryComboBox->lineEdit(), &QLineEdit::returnPressed, [this](){
     openViewerButton->animateClick();
   });

   framesCutLabel = new QLabel("", this);

   labelsTable = new QTableWidget(0, 2);
   labelsTable->setSelectionBehavior(QAbstractItemView::SelectRows);

   QStringList labels;
   labels << tr("Filename") << tr("Size");
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
   _scrollArea->setVisible(false);
   _scrollArea->setVisible(true);

   classCountTable = new QTableWidget(0, 2);
   QStringList classCountTableLabels;
   classCountTableLabels << tr("Class color") << tr("Count");
   classCountTable->setHorizontalHeaderLabels(classCountTableLabels);
   classCountTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
   classCountTable->verticalHeader()->hide();
   classCountTable->setShowGrid(false);

   QGridLayout *mainLayout = new QGridLayout(this);
   mainLayout->addWidget(new QLabel(tr("Video file:")), 0, 0);
   mainLayout->addWidget(imagesDirectoryComboBox, 0, 1);
   mainLayout->addWidget(imagesDirectoryButton, 0, 2);
   mainLayout->addWidget(new QLabel(tr("Output directory:")), 1, 0);
   mainLayout->addWidget(labelsDirectoryComboBox, 1, 1);
   mainLayout->addWidget(labelsDirectoryButton, 1, 2);
   mainLayout->addWidget(labelsTable, 2, 0, 1, 3);
   mainLayout->addWidget(classCountTable, 3, 0, 1, 3);
   mainLayout->addWidget(framesCutLabel, 4, 0, 1, 3);
   mainLayout->addWidget(openViewerButton, 5, 2);
   mainLayout->addWidget(_scrollArea, 0, 4, 5, 1);

   connect(new QShortcut(QKeySequence::Quit, this), &QShortcut::activated, qApp, &QApplication::quit);

   resize(QGuiApplication::primaryScreen()->availableSize() * 3 / 5);
}

static void updateComboBox(QComboBox *comboBox)
{
   if (comboBox->findText(comboBox->currentText()) == -1)
   {
      comboBox->addItem(comboBox->currentText());
   }
}

void OpenDatasetsDialog::openViewer()
{
   labelsTable->setRowCount(0);

   auto imagesDirectoryPath = QDir::cleanPath(imagesDirectoryComboBox->currentText()).toStdString();
   auto labelsDirectoryPath = QDir::cleanPath(labelsDirectoryComboBox->currentText()).toStdString();

   std::vector<fs::directory_entry> imageList(fs::directory_iterator{imagesDirectoryPath}, fs::directory_iterator{});

   QProgressDialog progressDialog(this);
   progressDialog.setCancelButtonText(tr("&Cancel"));
   progressDialog.setRange(0, imageList.size());
   progressDialog.setWindowTitle(tr("Counting labels"));

   std::map<cv::Vec3b, std::vector<cv::Rect>> allLabels;
   std::set<cv::Vec3b> colorSet;
   auto currentLabel = 0;
   for (auto file : imageList)
   {
     _dataset.emplace_back(file.path().string(), labelsDirectoryPath + "/" + file.path().filename().string());
     auto currentLabels = calculateLabels(colorSet, _dataset.back().second);
     for (auto const& rects : currentLabels)
     {
       allLabels[rects.first].insert(allLabels[rects.first].end(), rects.second.cbegin(), rects.second.cend());
     }
     auto filePathQ = QString::fromStdString(_dataset.back().first);
     const QString toolTip = QDir::toNativeSeparators(filePathQ);
     const QString relativePath = QDir::toNativeSeparators(currentDir.relativeFilePath(filePathQ));
     const qint64 size = QFileInfo(filePathQ).size();
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
     labelsTable->setItem(row, 0, fileNameItem);
     labelsTable->setItem(row, 1, sizeItem);

     progressDialog.setValue(currentLabel);
     progressDialog.setLabelText(tr("Processed label number %1 of %n...", nullptr, imageList.size()).arg(currentLabel++));
     QCoreApplication::processEvents();

     if (progressDialog.wasCanceled())
     {
       break;
     }
   }

   for (auto const& classLabels : allLabels)
   {
     QTableWidgetItem *classColorItem = new QTableWidgetItem(QString::fromStdString(std::to_string(classLabels.first[0]) + " " +
                                                                                    std::to_string(classLabels.first[1]) + " " +
                                                                                    std::to_string(classLabels.first[2])));
     //classColorItem->setData(absoluteFileNameRole, QVariant(filePathQ));
     //classColorItem->setToolTip(toolTip);
     classColorItem->setFlags(classColorItem->flags() ^ Qt::ItemIsEditable);
     QTableWidgetItem *countItem = new QTableWidgetItem(QString::fromStdString(std::to_string(classLabels.second.size())));
     //countItem->setData(absoluteFileNameRole, QVariant(filePathQ));
     //countItem->setToolTip(toolTip);
     countItem->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
     countItem->setFlags(countItem->flags() ^ Qt::ItemIsEditable);
     int row = classCountTable->rowCount();
     classCountTable->insertRow(row);
     classCountTable->setItem(row, 0, classColorItem);
     classCountTable->setItem(row, 1, countItem);
   }
//   framesCutLabel->setText(tr("%n file(s) found (Double click on a file to open it)", nullptr, currentFrame));
//   framesCutLabel->setWordWrap(true);
}

QComboBox *OpenDatasetsDialog::createComboBox(const QString &text)
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
  cv::Mat labelsImage = cv::imread(_dataset[row].second);
  cv::addWeighted(frame, 1.0, labelsImage, 0.5, 0.0, frame);
  cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

  image = QImage((uchar*)frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
  _labelsViewLabel->setPixmap(QPixmap::fromImage(image));
  _labelsViewLabel->adjustSize();
}
