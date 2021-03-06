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

class StartValidatingDialog : public QDialog
{
Q_OBJECT

public:
    StartValidatingDialog(std::string const& projectFileName, QWidget *parent = nullptr);

private:
    void validatingProcess();

public:
    std::string _projectFileName;
    boost::property_tree::ptree _pt;
};
