#include "MainWindow.hpp"

#include "NewTrainingProjectDialog.hpp"
#include "OpenDatasetsDialog.hpp"

#include <QtWidgets>

MainWindow::MainWindow(QWidget *parent)
  : QMainWindow(parent)
{
  createActions();
  createMenus();

  setWindowTitle(tr("UNet2D training tool"));
  setMinimumSize(160, 160);
}

void MainWindow::createActions()
{
  newSessionAct = new QAction(tr("&New training project..."), this);
  newSessionAct->setShortcuts(QKeySequence::New);
  newSessionAct->setStatusTip(tr("Create a new training project"));
  connect(newSessionAct, &QAction::triggered, [this](){
    NewTrainingProjectDialog newTrainingProjectDialog(false, this);
    newTrainingProjectDialog.show();
    if (newTrainingProjectDialog.exec() == QDialog::Accepted)
    {
      OpenDatasetsDialog openDatasetsDialog(newTrainingProjectDialog._projectFileName.toStdString(), this);
      openDatasetsDialog.show();
      openDatasetsDialog.exec();
    }
  });

  openSessionAct = new QAction(tr("&Open training project..."), this);
  openSessionAct->setShortcuts(QKeySequence::New);
  openSessionAct->setStatusTip(tr("Open a new training project"));
  connect(openSessionAct, &QAction::triggered, [this](){
    NewTrainingProjectDialog newTrainingProjectDialog(true, this);
    newTrainingProjectDialog.show();
    if (newTrainingProjectDialog.exec() == QDialog::Accepted)
    {
      OpenDatasetsDialog openDatasetsDialog(newTrainingProjectDialog._projectFileName.toStdString(), this);
      openDatasetsDialog.show();
      openDatasetsDialog.exec();
    }
  });

  exitAct = new QAction(tr("E&xit"), this);
  exitAct->setShortcuts(QKeySequence::Quit);
  exitAct->setStatusTip(tr("Exit the application"));
  connect(exitAct, &QAction::triggered, this, &QWidget::close);

  aboutAct = new QAction(tr("&About"), this);
  aboutAct->setStatusTip(tr("Show the application's About box"));
  connect(aboutAct, &QAction::triggered, [this]() {
    QMessageBox::about(this, tr("About UNetTrainingTool"), tr("The <b>UNetTrainingTool</b> simplifies prepearing and starting training custom UNet2D DNN."));
  });

  aboutQtAct = new QAction(tr("About &Qt"), this);
  aboutQtAct->setStatusTip(tr("Show the Qt library's About box"));
  connect(aboutQtAct, &QAction::triggered, qApp, &QApplication::aboutQt);
}

void MainWindow::createMenus()
{
  fileMenu = menuBar()->addMenu(tr("&File"));
  fileMenu->addAction(newSessionAct);
  fileMenu->addAction(openSessionAct);
  fileMenu->addSeparator();
  fileMenu->addAction(exitAct);

  helpMenu = menuBar()->addMenu(tr("&Help"));
  helpMenu->addAction(aboutAct);
  helpMenu->addAction(aboutQtAct);
}