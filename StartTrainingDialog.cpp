#include "StartTrainingDialog.hpp"

#include <QtWidgets>

#include <boost/property_tree/json_parser.hpp>

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

  auto mainLayout = new QGridLayout;
  mainLayout->addWidget(new QLabel(tr("Input channels count:")), 0, 0);
  mainLayout->addWidget(inputChannelsComboBox, 0, 1);
  mainLayout->addWidget(new QLabel(tr("Output channels count:")), 1, 0);
  mainLayout->addWidget(outputChannelsSpinBox, 1, 1);
  mainLayout->addWidget(new QLabel(tr("Levels count:")), 2, 0);
  mainLayout->addWidget(levelsCountSpinBox, 2, 1);
  mainLayout->addWidget(new QLabel(tr("Initial features count:")), 3, 0);
  mainLayout->addWidget(featuresCountPowSpinBox, 3, 1);

  setLayout(mainLayout);
}

