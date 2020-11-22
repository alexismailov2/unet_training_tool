#include "NewTrainingProjectDialog.hpp"

#include <QtWidgets>

#include <boost/property_tree/json_parser.hpp>

#ifdef _MSC_VER
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include <iostream>

namespace bp = boost::property_tree;

namespace {
void appendNewDataset(bp::ptree& tree, std::string const& images, std::string const& annotations)
{
  bp::ptree datasetItem;
  datasetItem.put("images", images);
  datasetItem.put("annotations", annotations);

  auto datasets = tree.get_child_optional("datasets");
  if (!datasets.has_value())
  {
    datasets = tree.put_child("datasets", bp::ptree{});
  }
  datasets.get().push_back(bp::ptree::value_type("", datasetItem));
}

void removeSelectedDataset(bp::ptree& tree, std::string const& annotationForDeletion)
{
  auto datasets = tree.get_child_optional("datasets");
  if (datasets.has_value())
  {
    auto result = std::find_if(datasets.get().begin(), datasets.get().end(), [&](bp::ptree::value_type const& element) {
      auto const currentAnnotation = element.second.get<std::string>("annotations");
      return currentAnnotation == annotationForDeletion;
    });
    if (result != datasets.get().end())
    {
      datasets.get().erase(result);
    }
  }
}
} /// end namespace anonymous

NewTrainingProjectDialog::NewTrainingProjectDialog(QWidget *parent)
  : QDialog(parent)
{
  _projectFileName = QFileDialog::getSaveFileName(this, tr("Select project file"),
                                                  ".",
                                                  tr("Json (*.json)"));
  if (_projectFileName.isEmpty())
  {
    close();
    deleteLater();
    return;
  }
  bp::write_json(_projectFileName.toStdString(), _pt);

  _datasetListWidget = new QListWidget(this);
  _addDatasetButton = new QPushButton(tr("&Add dataset"));
  connect(_addDatasetButton, &QPushButton::clicked,this, [&](){
    QString dir = QFileDialog::getExistingDirectory(this, tr("Open directory with \"labelme\" annotations"),
                                                    ".",
                                                    QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (dir.isEmpty())
    {
      return;
    }

    std::string imagePath;
    for (auto item : fs::directory_iterator(dir.toStdString()))
    {
      auto const fullPath = item.path().string();
      auto const extention = item.path().filename().extension().string();
      if (extention == ".json")
      {
        try
        {
          bp::ptree annotation;
          bp::read_json(fullPath, annotation);
          auto imagePathOpt = annotation.get_optional<std::string>("imagePath");
          if (imagePathOpt.has_value())
          {
            auto imageDirPath = imagePathOpt.get().substr(0, imagePathOpt.get().find_last_of('\\'));
            std::replace(imageDirPath.begin(), imageDirPath.end(), '\\', '/');
            auto const lastSlashPos = fullPath.find_last_of('/');
            auto const directory = fullPath.substr(0, lastSlashPos + 1);
            auto imagesRelativeDirectory = QDir::toNativeSeparators(QString::fromStdString(imageDirPath)).toStdString();
            imagePath = QDir::toNativeSeparators(QString::fromStdString(directory)).toStdString() + imagesRelativeDirectory;
            break;
          }
        }
        catch(bp::json_parser_error const&)
        {
          continue;
        }
      }
    }
    appendNewDataset(_pt, imagePath, dir.toStdString());
    bp::write_json(_projectFileName.toStdString(), _pt);
    new QListWidgetItem(dir, _datasetListWidget);
  });

  _removeDatasetButton = new QPushButton(tr("&Remove dataset"));
  _removeDatasetButton->setEnabled(false);
  connect(_removeDatasetButton, &QPushButton::clicked,this, [&](){
    if (_datasetListWidget->currentRow() != -1)
    {
      removeSelectedDataset(_pt, _datasetListWidget->currentItem()->text().toStdString());
      boost::property_tree::write_json(_projectFileName.toStdString(), _pt);
      qDeleteAll(_datasetListWidget->selectedItems());
    }
  });

  connect(_datasetListWidget, &QListWidget::currentRowChanged, [&](int row){
    _removeDatasetButton->setEnabled((row != -1));
  });

  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

  connect(buttonBox, SIGNAL(accepted()), this, SLOT(accept()));
  connect(buttonBox, SIGNAL(rejected()), this, SLOT(reject()));

  QVBoxLayout* buttonLayout1 = new QVBoxLayout;
  buttonLayout1->addWidget(_addDatasetButton);
  buttonLayout1->addWidget(_removeDatasetButton);
  buttonLayout1->addStretch();

  QGridLayout *mainLayout = new QGridLayout;
  mainLayout->addWidget(new QLabel(tr("Datasets:")), 0, 0, Qt::AlignTop);
  mainLayout->addWidget(_datasetListWidget, 0, 1);
  mainLayout->addLayout(buttonLayout1, 0, 2);
  mainLayout->addWidget(buttonBox, 1, 2);

  setLayout(mainLayout);
  setWindowTitle(tr("Create new training project"));
}