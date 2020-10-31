#ifndef UNETTRAININGTOOL_OPENDATASETSDIALOG_HPP
#define UNETTRAININGTOOL_OPENDATASETSDIALOG_HPP

#include <QDialog>
#include <QDir>

QT_BEGIN_NAMESPACE
class QComboBox;
class QLabel;
class QPushButton;
class QTableWidget;
class QTableWidgetItem;
class QScrollArea;
QT_END_NAMESPACE

class OpenDatasetsDialog : public QDialog
{
Q_OBJECT

public:
    OpenDatasetsDialog(QWidget *parent = nullptr);

private slots:
    void openViewer();
    void createDatasetLists();
    void openDatasetItem(int row, int, int, int);

private:
    QComboBox *createComboBox(const QString &text = QString());

    QComboBox *imagesDirectoryComboBox;
    QComboBox *labelsDirectoryComboBox;
    QLabel *framesCutLabel;
    QPushButton *openViewerButton;
    QPushButton* createDatasetButton;
    QTableWidget *labelsTable;
    QTableWidget *classCountTable;
    std::vector<std::pair<std::string, std::string>> _dataset;
    QDir currentDir;

    QLabel* _labelsViewLabel;
    QScrollArea* _scrollArea;
    QImage image;
};

#endif //UNETTRAININGTOOL_OPENDATASETSDIALOG_HPP
