#pragma once

#include <QDialog>
#include <QDir>
#include <QMap>

#include <boost/property_tree/ptree.hpp>

QT_BEGIN_NAMESPACE
class QPushButton;
class QLabel;
class QListWidget;
QT_END_NAMESPACE

class NewTrainingProjectDialog : public QDialog
{
  Q_OBJECT

public:
  NewTrainingProjectDialog(bool isOpen, QWidget *parent = nullptr);

private:
  QPushButton* _addDatasetButton{};
  QListWidget* _datasetListWidget{};
  QPushButton* _removeDatasetButton{};
  boost::property_tree::ptree _pt;

public:
  QString _projectFileName;
};


