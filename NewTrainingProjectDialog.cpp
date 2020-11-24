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
bool appendNewDataset(bp::ptree& tree, std::string const& images, std::string const& annotations)
{
  bp::ptree datasetItem;
  datasetItem.put("images", images);
  datasetItem.put("annotations", annotations);

  auto datasets = tree.get_child_optional("datasets");
  if (!datasets.has_value())
  {
    datasets = tree.put_child("datasets", bp::ptree{});
  }
  auto result = std::find_if(datasets.get().begin(), datasets.get().end(), [&](bp::ptree::value_type const& element) {
    auto const currentAnnotation = element.second.get<std::string>("annotations");
    return currentAnnotation == annotations;
  });
  if (result != datasets.get().end())
  {
    return false;
  }
  datasets.get().push_back(bp::ptree::value_type("", datasetItem));
  return true;
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

auto createComboBox(QStringList const& itemsList = {}, QWidget* parent = nullptr) -> QComboBox*
{
  auto comboBox = new QComboBox{parent};
  comboBox->setEditable(true);
  comboBox->addItems(itemsList);
  comboBox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
  return comboBox;
}
} /// end namespace anonymous

NewTrainingProjectDialog::NewTrainingProjectDialog(bool isOpen, QWidget *parent)
  : QDialog(parent)
{
  _projectFileName = isOpen ? QFileDialog::getOpenFileName(this, tr("Select project file"),".",tr("Json (*.json)"))
                            : QFileDialog::getSaveFileName(this, tr("Select project file"),".",tr("Json (*.json)"));
  if (_projectFileName.isEmpty())
  {
    close();
    deleteLater();
    return;
  }

  if (isOpen)
  {
    bp::read_json(_projectFileName.toStdString(), _pt);
  }
  else
  {
    bp::write_json(_projectFileName.toStdString(), _pt);
  }

  _datasetListWidget = new QListWidget(this);
  auto datasets = _pt.get_child_optional("datasets");
  if (datasets.has_value())
  {
    for (auto& item : datasets.get())
    {
      auto dir = item.second.get_optional<std::string>("annotations");
      new QListWidgetItem(QString::fromStdString(dir.get()), _datasetListWidget);
    }
  }
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
    if (!appendNewDataset(_pt, imagePath, dir.toStdString()))
    {
      QMessageBox msgBox;
      msgBox.setText(QString::fromStdString(imagePath + " is already exists!"));
      msgBox.exec();
    }
    else
    {
      bp::write_json(_projectFileName.toStdString(), _pt);
      new QListWidgetItem(dir, _datasetListWidget);
    }
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

//  mainLayout->addWidget(new QLabel(tr("Input channels count:")), 2, 0);
//  mainLayout->addWidget(inputChannelsComboBox, 2, 1);
//  mainLayout->addWidget(new QLabel(tr("Output channels count:")), 3, 0);
//  mainLayout->addWidget(outputChannelsSpinBox, 3, 1);
//  mainLayout->addWidget(new QLabel(tr("Levels count:")), 4, 0);
//  mainLayout->addWidget(levelsCountSpinBox, 4, 1);
//  mainLayout->addWidget(new QLabel(tr("Initial features count:")), 5, 0);
//  mainLayout->addWidget(featuresCountPowSpinBox, 5, 1);

  setLayout(mainLayout);
  setWindowTitle(tr("Create new training project"));
}