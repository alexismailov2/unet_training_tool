#include <QApplication>
#include <QtCore/QTranslator>
#include <OpenDatasetsDialog.hpp>

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);
  OpenDatasetsDialog d;
  d.exec();
  return a.exec();
}
