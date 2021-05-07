#pragma once

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui { class CableDeffectsFinderWindow; }
class QAction;
class QActionGroup;
class QLabel;
class QMenu;
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
Q_OBJECT

public:
  MainWindow(QWidget *parent = nullptr);

private:
  void createActions();
  void createMenus();

  QMenu* fileMenu{};
  QMenu* helpMenu{};
  QAction* newSessionAct{};
  QAction* openSessionAct{};
  QAction* validSessionAct{};
  QAction* exitAct{};
  QAction* aboutAct{};
  QAction* aboutQtAct{};
};
