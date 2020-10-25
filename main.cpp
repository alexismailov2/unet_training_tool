#include <QApplication>
#include <QtCore/QTranslator>
#include <OpenDatasetsDialog.hpp>

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);
  //QLocale::setDefault(QLocale(QLocale::Russian, QLocale::RussianFederation));
  QTranslator qtTranslator;
  qtTranslator.load("CableDeffectsFinder_ru_RU");
  a.installTranslator(&qtTranslator);
  //QTranslator myappTranslator;
  //myappTranslator.load(QLocale("ru_RU"), QLatin1String("/home/oleksandr_ismailov/WORK/Upwork/CableDeffectsFinder/CableDeffectsFinder_ru_RU.ts"));
  //a.installTranslator(&myappTranslator);

  OpenDatasetsDialog d;
  d.exec();
  return a.exec();
}
