#include "StartValidatingDialog.hpp"
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

StartValidatingDialog::StartValidatingDialog(std::string const& projectFileName, QWidget* parent)
   : QDialog(parent)
   , _projectFileName{projectFileName}
{
    setWindowTitle(tr("Start validating dialog"));

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
    auto weightsFilePathButton = new QPushButton(tr("Weights path"), this);
    connect(weightsFilePathButton, &QAbstractButton::clicked, [this](){
        auto weightsFilePath = QFileDialog::getOpenFileName(this, tr("Select weights file"),".",tr("Darknet weights (*.weights)")).toStdString();
        _pt.put<std::string>("UNet.weightsFilePath", weightsFilePath);
        boost::property_tree::write_json(_projectFileName, _pt);
    });
    auto modelFilePathButton = new QPushButton(tr("Model path"), this);
    connect(modelFilePathButton, &QAbstractButton::clicked, [this](){
        auto modelFilePath = QFileDialog::getOpenFileName(this, tr("Select model file"),".",tr("Darknet model (*.cfg)")).toStdString();
        _pt.put<std::string>("UNet.modelFilePath", modelFilePath);
        boost::property_tree::write_json(_projectFileName, _pt);
    });
    auto validDatasetPathButton = new QPushButton(tr("Valid dataset path"), this);
    connect(validDatasetPathButton, &QAbstractButton::clicked, [this](){
        auto validDatasetPath = QFileDialog::getExistingDirectory(this, tr("Open dataset directory"), "", QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks).toStdString();
        _pt.put<std::string>("UNet.validDatasetPath", validDatasetPath);
        boost::property_tree::write_json(_projectFileName, _pt);
    });
    auto isEvalCheckBox = new QCheckBox(tr("Evaluation only"), this);
    connect(isEvalCheckBox, &QCheckBox::clicked, [this](bool isChecked){
        _pt.put<bool>("UNet.evaluationOnly", isChecked);
        boost::property_tree::write_json(_projectFileName, _pt);
    });
    auto startTrainingButton = new QPushButton(tr("Start validating"), this);
    connect(startTrainingButton, &QAbstractButton::clicked, [this](){
        validatingProcess();
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
    mainLayout->addWidget(new QLabel(tr("Model file path:")), 7, 0);
    mainLayout->addWidget(modelFilePathButton, 7, 1);
    mainLayout->addWidget(new QLabel(tr("Weights file path:")), 8, 0);
    mainLayout->addWidget(weightsFilePathButton, 8, 1);
    mainLayout->addWidget(new QLabel(tr("Valid dataset path:")), 9, 0);
    mainLayout->addWidget(validDatasetPathButton, 9, 1);
    mainLayout->addWidget(isEvalCheckBox, 10, 0);
    mainLayout->addWidget(startTrainingButton, 11, 0);

    setLayout(mainLayout);
}

void StartValidatingDialog::validatingProcess()
{
    auto modelFilePath = _pt.get<std::string>("UNet.modelFilePath", "");
    auto weightsFilePath = _pt.get<std::string>("UNet.weightsFilePath", "");
    auto convertedDatasetDir = _pt.get<std::string>("UNet.validDatasetPath", "");
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
    params["--eval"] = {"yes"};
    params["--epochs"] = {std::to_string(_pt.get<uint32_t>("UNet.epochsCount"))};
    fs::create_directories(modelFilePath + "_checkpoints");
    params["--checkpoints-output"] = {modelFilePath + "_checkpoints"};
    params["--train-directories"] = {convertedDatasetDir + "/imagesT/",convertedDatasetDir + "/masksT/"};
    params["--valid-directories"] = {convertedDatasetDir + "/imagesV/",convertedDatasetDir + "/masksV/"};
    params["--model-darknet"] = {modelFilePath, weightsFilePath};
    params["--size-downscaled"] = {"0","0"};
    params["--grayscale"] = {(_pt.get<uint32_t>("UNet.inputChannels") == 1) ? "yes" : "no"};
    runOpts(params);
}
