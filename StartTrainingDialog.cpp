#include "StartTrainingDialog.hpp"
#include "ProjectFile.hpp"

#include <third_party/UNetDarknetTorch/include/UNet/TrainUnet2D.hpp>

#include <QtWidgets>

#include <boost/property_tree/json_parser.hpp>

#ifdef _MSC_VER
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

namespace bp = boost::property_tree;

namespace {
auto createComboBox(QStringList const& itemsList = {}, QWidget* parent = nullptr) -> QComboBox*
{
  auto comboBox = new QComboBox{parent};
  comboBox->setEditable(true);
  comboBox->addItems(itemsList);
  comboBox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
  return comboBox;
}
} /// end namespace anonymous

StartTrainingDialog::StartTrainingDialog(std::string const& projectFileName, QWidget* parent)
  : QDialog(parent)
  , _projectFileName{projectFileName}
{
  setWindowTitle(tr("Start training dialog"));

  boost::property_tree::read_json(_projectFileName, _pt);

  if (!_pt.get_optional<uint32_t>("UNet.inputChannels").is_initialized())
  {
    _pt.put<uint32_t>("UNet.inputChannels", 1);
  }
  if (!_pt.get_optional<uint32_t>("UNet.outputChannels").is_initialized())
  {
    _pt.put<uint32_t>("UNet.outputChannels", 1);
  }
  if (!_pt.get_optional<uint32_t>("UNet.layersCount").is_initialized())
  {
    _pt.put<uint32_t>("UNet.layersCount", 3);
  }
  if (!_pt.get_optional<uint32_t>("UNet.featuresCount").is_initialized())
  {
    _pt.put<uint32_t>("UNet.featuresCount", 3);
  }
  boost::property_tree::write_json(_projectFileName, _pt);

  auto inputChannelsComboBox = createComboBox({"1", "3"},this);
  connect(inputChannelsComboBox, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [this](int index) {
    _pt.put<uint32_t>("UNet.inputChannels", index == 0 ? 1 : 3);
    boost::property_tree::write_json(_projectFileName, _pt);
  });

  auto outputChannelsSpinBox = new QSpinBox{this};
  outputChannelsSpinBox->setSingleStep(1);
  outputChannelsSpinBox->setValue(_pt.get<uint32_t>("UNet.outputChannels", 1));
  outputChannelsSpinBox->setMinimum(1);
  outputChannelsSpinBox->setMaximum(256);
  connect(outputChannelsSpinBox, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), [this](int value) {
    _pt.put<uint32_t>("UNet.outputChannels", value);
    boost::property_tree::write_json(_projectFileName, _pt);
  });

  auto levelsCountSpinBox = new QSpinBox{this};
  levelsCountSpinBox->setSingleStep(1);
  levelsCountSpinBox->setValue(_pt.get<uint32_t>("UNet.layersCount", 1));
  outputChannelsSpinBox->setMinimum(1);
  outputChannelsSpinBox->setMaximum(16);
  connect(levelsCountSpinBox, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), [this](int value) {
    _pt.put<uint32_t>("UNet.layersCount", value);
    boost::property_tree::write_json(_projectFileName, _pt);
  });

  auto featuresCountPowSpinBox = new QSpinBox{this};
  featuresCountPowSpinBox->setSingleStep(1);
  featuresCountPowSpinBox->setValue(std::log2(_pt.get<uint32_t>("UNet.featuresCount", 8)));
  connect(featuresCountPowSpinBox, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), [this](int value) {
    _pt.put<uint32_t>("UNet.featuresCount", (1 << value));
    boost::property_tree::write_json(_projectFileName, _pt);
  });

  auto heightDownscaleSpinBox = new QSpinBox{this};
  heightDownscaleSpinBox->setSingleStep(1);
  heightDownscaleSpinBox->setValue(_pt.get<uint32_t>("UNet.heightDownscale", 1));
  heightDownscaleSpinBox->setMinimum(1);
  heightDownscaleSpinBox->setMaximum(16);
  connect(heightDownscaleSpinBox, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), [this](int value) {
    _pt.put<uint32_t>("UNet.heightDownscale", value);
    boost::property_tree::write_json(_projectFileName, _pt);
  });

  auto widthDownscaleSpinBox = new QSpinBox{this};
  widthDownscaleSpinBox->setSingleStep(1);
  widthDownscaleSpinBox->setValue(_pt.get<uint32_t>("UNet.widthDownscale", 1));
  widthDownscaleSpinBox->setMinimum(1);
  widthDownscaleSpinBox->setMaximum(16);
  connect(widthDownscaleSpinBox, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), [this](int value) {
    _pt.put<uint32_t>("UNet.widthDownscale", value);
    boost::property_tree::write_json(_projectFileName, _pt);
  });

  auto epochsCountSpinBox = new QSpinBox{this};
  epochsCountSpinBox->setSingleStep(1);
  epochsCountSpinBox->setValue(_pt.get<uint32_t>("UNet.epochsCount", 200));
  epochsCountSpinBox->setMinimum(1);
  epochsCountSpinBox->setMaximum(100000);
  connect(epochsCountSpinBox, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), [this](int value) {
    _pt.put<uint32_t>("UNet.epochsCount", value);
    boost::property_tree::write_json(_projectFileName, _pt);
  });

  auto startTrainingButton = new QPushButton(tr("Start training"), this);
  connect(startTrainingButton, &QAbstractButton::clicked, [this](){
    trainingProcess();
  });
  auto weightsFilePathButton = new QPushButton(tr("Weights path"), this);
  connect(weightsFilePathButton, &QAbstractButton::clicked, [this](){
      auto weightsFilePath = QFileDialog::getOpenFileName(this, tr("Select weights file"),".",tr("Darknet weights (*.weights)")).toStdString();
      _pt.put<std::string>("UNet.weightsFilePath", weightsFilePath);
      boost::property_tree::write_json(_projectFileName, _pt);
  });
  auto isEvalCheckBox = new QCheckBox(tr("Evaluation only"), this);
  connect(isEvalCheckBox, &QCheckBox::clicked, [this](bool isChecked){
      _pt.put<bool>("UNet.evaluationOnly", isChecked);
      boost::property_tree::write_json(_projectFileName, _pt);
  });
  auto mainLayout = new QGridLayout;
  mainLayout->addWidget(new QLabel(tr("Input channels count:")), 0, 0);
  mainLayout->addWidget(inputChannelsComboBox, 0, 1);
  mainLayout->addWidget(new QLabel(tr("Output channels count:")), 1, 0);
  mainLayout->addWidget(outputChannelsSpinBox, 1, 1);
  mainLayout->addWidget(new QLabel(tr("Levels count:")), 2, 0);
  mainLayout->addWidget(levelsCountSpinBox, 2, 1);
  mainLayout->addWidget(new QLabel(tr("Initial features count:")), 3, 0);
  mainLayout->addWidget(featuresCountPowSpinBox, 3, 1);
  mainLayout->addWidget(new QLabel(tr("Width downscale:")), 4, 0);
  mainLayout->addWidget(widthDownscaleSpinBox, 4, 1);
  mainLayout->addWidget(new QLabel(tr("Height downscale:")), 5, 0);
  mainLayout->addWidget(heightDownscaleSpinBox, 5, 1);
  mainLayout->addWidget(new QLabel(tr("Epochs count:")), 6, 0);
  mainLayout->addWidget(epochsCountSpinBox, 6, 1);
  mainLayout->addWidget(new QLabel(tr("Weights file path:")), 7, 0);
  mainLayout->addWidget(weightsFilePathButton, 7, 1);
  mainLayout->addWidget(isEvalCheckBox, 8, 0);
  mainLayout->addWidget(startTrainingButton, 9, 0);

  setLayout(mainLayout);
}

void StartTrainingDialog::trainingProcess()
{
  auto datasetFolderPathes = _pt.get_child_optional("datasets");
  if (!datasetFolderPathes.is_initialized())
  {
    QMessageBox msgBox;
    msgBox.setText("Could not be found any dataset folder in the project file!");
    msgBox.exec();
    return;
  }

  QString dir = QFileDialog::getExistingDirectory(this, tr("Open directory for saving generated network"),
                                                  ".",
                                                  QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
  if (dir.isEmpty())
  {
    QMessageBox msgBox;
    msgBox.setText("Should be selected folder for saving model!");
    msgBox.exec();
    return;
  }
  auto modelFilePath = dir.toStdString() + "/unet_" +
                       _pt.get<std::string>("UNet.inputChannels", "1") + "c" +
                       _pt.get<std::string>("UNet.outputChannels", "7") + "cl" +
                       _pt.get<std::string>("UNet.layersCount", "4") + "l" +
                       _pt.get<std::string>("UNet.featuresCount", "5") + "f" + ".cfg";
  auto weightsFilePath = _pt.get<std::string>("UNet.weightsFilePath", "");
  _pt.put<std::string>("UNet.modelFilePath", modelFilePath);
  boost::property_tree::write_json(_projectFileName, _pt);
  runOpts({{std::string("--generate-custom-unet"), {
                                                     _pt.get<std::string>("UNet.inputChannels"),
                                                     _pt.get<std::string>("UNet.outputChannels"),
                                                     _pt.get<std::string>("UNet.layersCount"),
                                                     _pt.get<std::string>("UNet.featuresCount"),
                                                     dir.toStdString()}}});

  auto convertedDatasetDir = QFileDialog::getExistingDirectory(this, tr("Open directory for saving converted dataset"),
                                                               ".",
                                                               QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
  if (convertedDatasetDir.isEmpty())
  {
    QMessageBox msgBox;
    msgBox.setText("Canceled!");
    msgBox.exec();
    return;
  }
#if 1
  fs::create_directories(convertedDatasetDir.toStdString() + "/masksT");
  fs::create_directories(convertedDatasetDir.toStdString() + "/imagesT");
  fs::create_directories(convertedDatasetDir.toStdString() + "/masksV");
  fs::create_directories(convertedDatasetDir.toStdString() + "/imagesV");

  auto const isClahe = false;
  auto const heightDownscale = _pt.get<uint32_t>("UNet.heightDownscale");
  auto const widthDownscale = _pt.get<uint32_t>("UNet.widthDownscale");
  auto const initialFeatureCount = _pt.get<uint32_t>("UNet.featuresCount");
  auto colorToClass = ProjectFile::loadColors(_pt);

  /// Getting whole list
  std::vector<std::pair<std::string, std::string>> wholeDatasetList;
  for (auto const& datasetFolderPath : datasetFolderPathes.get())
  {
      auto annotations = datasetFolderPath.second.get_optional<std::string>("annotations");
      auto images = datasetFolderPath.second.get_optional<std::string>("images");
      if (!annotations.is_initialized() || !images.is_initialized())
      {
          continue;
      }
      for (auto const& file : fs::directory_iterator{images.get()})
      {
          if (fs::is_directory(file))
          {
              continue;
          }
          auto filename = file.path().filename().string();
          filename = filename.substr(0, filename.size() - 4);
          wholeDatasetList.emplace_back(std::make_pair(file.path().string(), annotations.get() + "/" + filename + ".json"));
      }
  }
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(wholeDatasetList.begin(), wholeDatasetList.end(), g);
  ////

  QProgressDialog progressDialog(this);
  progressDialog.setCancelButtonText(tr("&Cancel"));
  progressDialog.setRange(0, wholeDatasetList.size());
  progressDialog.setWindowTitle(tr("Counting labels"));

  auto currentLabel = 0;
  for (auto const& datasetItem : wholeDatasetList)
  {
      cv::Mat mask = ConvertPolygonsToMask(datasetItem.second, colorToClass);
      if (mask.empty())
      {
          QMessageBox msgBox;
          msgBox.setText(QString("Could not be gotten mask from annotation file: ") + QString::fromStdString(datasetItem.second));
          msgBox.exec();
          continue;
      }
      cv::resize(mask, mask, cv::Size((mask.cols) / widthDownscale, (((uint32_t)mask.rows) / heightDownscale)), 0, 0, cv::INTER_NEAREST);
      auto truncatedCols = mask.cols & (~(initialFeatureCount - 1));
      auto truncatedRows = mask.rows & (~(initialFeatureCount - 1));
      auto const offsetX = 0; //256 + 128;
      auto const sizeSubX = 0; //512;
      auto roi = cv::Rect(((mask.cols - truncatedCols) / 2) + offsetX, (mask.rows - truncatedRows) / 2, truncatedCols - sizeSubX, truncatedRows);

      auto isTraining = currentLabel > wholeDatasetList.size() * 0.1f;
      cv::Mat maskCropped = roi.empty() ? mask : mask(roi);
      auto filename = fs::path(datasetItem.first).filename().replace_extension("png").string();
      cv::imwrite(convertedDatasetDir.toStdString() + "/masks" + (isTraining ? "T/" : "V/") + filename, maskCropped);

      cv::Mat image = cv::imread(datasetItem.first);
      cv::resize(image, image, cv::Size((image.cols / widthDownscale), (image.rows / heightDownscale)), 0, 0, cv::INTER_NEAREST);
      if (isClahe)
      {
          auto clahe = cv::createCLAHE();
          cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
          clahe->apply(image, image);
      }
      cv::Mat imageCropped = roi.empty() ? image : image(roi);
      cv::imwrite(convertedDatasetDir.toStdString() + "/images" + (isTraining ? "T/" : "V/") + filename, imageCropped);

      progressDialog.setValue(currentLabel);
      progressDialog.setLabelText(tr("Processed label number %1 of %n ...", nullptr, wholeDatasetList.size()).arg(currentLabel++));
      QCoreApplication::processEvents();

      if (progressDialog.wasCanceled())
      {
          break;
      }
  }
#if 0
  auto currentLabel = 0;
  for (auto const& datasetFolderPath : datasetFolderPathes.get())
  {
    auto annotations = datasetFolderPath.second.get_optional<std::string>("annotations");
    auto images = datasetFolderPath.second.get_optional<std::string>("images");
    if (!annotations.is_initialized() || !images.is_initialized())
    {
      continue;
    }
    auto const heightDownscale = 2;
    auto const widthDownscale = 1;
    auto const initialFeatureCount = _pt.get<uint32_t>("UNet.featuresCount");
    auto colorToClass = ProjectFile::loadColors(_pt);

    std::vector<fs::directory_entry> annotationsList(fs::directory_iterator{annotations.get()}, fs::directory_iterator{});
    for (auto const& file : annotationsList)
    {
      if (fs::is_directory(file))
      {
        continue;
      }
      cv::Mat mask = ConvertPolygonsToMask(file.path().string(), colorToClass);
      if (mask.empty())
      {
        QMessageBox msgBox;
        msgBox.setText(QString("Could not be gotten mask from annotation file: ") + QString::fromStdString(file.path().string()));
        msgBox.exec();
        continue;
      }
      cv::resize(mask, mask, cv::Size{(mask.cols / widthDownscale), (mask.rows / heightDownscale)}, 0, 0, cv::INTER_NEAREST);
      auto truncatedCols = mask.cols & (~(initialFeatureCount - 1));
      auto truncatedRows = mask.rows & (~(initialFeatureCount - 1));
      auto roi = cv::Rect{((mask.cols - truncatedCols) / 2) + 256 + 128, (mask.rows - truncatedRows) / 2, truncatedCols - 512, truncatedRows};
      auto filename = file.path().filename().string();
      filename = filename.substr(0, filename.size() - 4);
      cv::Mat maskCropped = roi.empty() ? mask : mask(roi);
      cv::imwrite(convertedDatasetDir.toStdString() + "/masks" + ((currentLabel & 1) ? "T/" : "V/") + filename + "png", maskCropped);

      const bool isClahe = true;
      cv::Mat image = cv::imread(images.get() + "/" + filename + "jpg");
      if (image.empty())
      {
        image = cv::imread(images.get() + "/" + filename + "png");
      }
      cv::resize(image, image, cv::Size{(image.cols / widthDownscale), (image.rows / heightDownscale)}, 0, 0, cv::INTER_NEAREST);
      if (isClahe)
      {
        auto clahe = cv::createCLAHE();
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        clahe->apply(image, image);
      }
      cv::Mat imageCropped = roi.empty() ? image : image(roi);
      cv::imwrite(convertedDatasetDir.toStdString() + "/images" + ((currentLabel & 1) ? "T/" : "V/") + filename + "png", imageCropped);

      progressDialog.setValue(currentLabel);
      progressDialog.setLabelText(tr("Processed label number %1 of %n ...", nullptr, annotationsList.size()).arg(currentLabel++));
      QCoreApplication::processEvents();

      if (progressDialog.wasCanceled())
      {
        break;
      }
    }
  }
#endif
#endif
  QMessageBox msgBox;
  msgBox.setText("Are you ready to train?");
  msgBox.exec();

  auto colorsToClassMap = ProjectFile::loadColors(_pt);
  std::map<std::string, std::vector<std::string>> params;
  for (auto const& colorToClass : colorsToClassMap)
  {
    params["--colors-to-class-map"].emplace_back(colorToClass.first);
    params["--colors-to-class-map"].emplace_back(std::to_string(colorToClass.second[2]));
    params["--colors-to-class-map"].emplace_back(std::to_string(colorToClass.second[1]));
    params["--colors-to-class-map"].emplace_back(std::to_string(colorToClass.second[0]));
    params["--selected-classes-and-thresholds"].emplace_back(colorToClass.first);
    params["--selected-classes-and-thresholds"].emplace_back("0.3");
    // TODO: tempoarry hack
    break;
  }
  params["--eval"] = {_pt.get<bool>("UNet.evaluationOnly") ? "yes" : "no"};
  params["--epochs"] = {std::to_string(_pt.get<uint32_t>("UNet.epochsCount"))};
  fs::create_directories(modelFilePath + "_checkpoints");
  params["--checkpoints-output"] = {modelFilePath + "_checkpoints"};
  params["--train-directories"] = {convertedDatasetDir.toStdString() + "/imagesT/",convertedDatasetDir.toStdString() + "/masksT/"};
  params["--valid-directories"] = {convertedDatasetDir.toStdString() + "/imagesV/",convertedDatasetDir.toStdString() + "/masksV/"};
  if (weightsFilePath.empty())
  {
      params["--model-darknet"] = {modelFilePath};
  }
  else
  {
      params["--model-darknet"] = {modelFilePath, weightsFilePath};
  }

  params["--size-downscaled"] = {"0","0"};
  params["--grayscale"] = {(_pt.get<uint32_t>("UNet.inputChannels") == 1) ? "yes" : "no"};
  runOpts(params);

  QMessageBox msgBox1;
  msgBox1.setText("Not implemented to be continued!");
  msgBox1.exec();
}
