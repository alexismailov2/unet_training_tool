cmake_minimum_required(VERSION 3.5)

project(unet-training-tool LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#add_subdirectory(third_party/???)

# QtCreator supports the following variables for Android, which are identical to qmake Android variables.
# Check http://doc.qt.io/qt-5/deployment-android.html for more information.
# They need to be set before the find_package(Qt5 ...) call.

#if(ANDROID)
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
#    if (ANDROID_ABI STREQUAL "armeabi-v7a")
#        set(ANDROID_EXTRA_LIBS
#            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libcrypto.so
#            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libssl.so)
#    endif()
#endif()

find_package(QT NAMES Qt6 Qt5 COMPONENTS Widgets LinguistTools REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets LinguistTools REQUIRED)
find_package(Boost REQUIRED)

if("${CUSTOM_OPENCV_BUILD_PATH}" STREQUAL "")
    find_package(OpenCV REQUIRED)
else()
    find_package(OpenCV REQUIRED PATHS ${CUSTOM_OPENCV_BUILD_PATH} NO_DEFAULT_PATH)
endif()

add_subdirectory(third_party)

set(TS_FILES UnetTrainingTool_ru_RU.ts)

include_directories(
    third_party/UNetDarknetTorch/include
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS})

if (ANDROID)
    add_library(${PROJECT_NAME} SHARED
        main.cpp
        MainWindow.hpp
        MainWindow.cpp
        NewTrainingProjectDialog.hpp
        NewTrainingProjectDialog.cpp
        OpenDatasetsDialog.cpp
        OpenDatasetsDialog.hpp
        StartTrainingDialog.hpp
        StartTrainingDialog.cpp
        ProjectFile.cpp
        ProjectFile.hpp
        #${TS_FILES}
        )
else ()
    add_executable(${PROJECT_NAME}
        main.cpp
        MainWindow.hpp
        MainWindow.cpp
        NewTrainingProjectDialog.hpp
        NewTrainingProjectDialog.cpp
        OpenDatasetsDialog.cpp
        OpenDatasetsDialog.hpp
        StartTrainingDialog.hpp
        StartTrainingDialog.cpp
        StartValidatingDialog.hpp
        StartValidatingDialog.cpp
        ProjectFile.cpp
        ProjectFile.hpp
        #${TS_FILES}
        )
endif ()

set(STD_FILESYSTEM)
if (NOT APPLE AND NOT MSVC)
    set(STD_FILESYSTEM stdc++fs)
endif ()
message(STATUS STD_FILESYSTEM=${STD_FILESYSTEM})

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    ${OpenCV_LIBS}
    ${Boost_LIBS}
    ${STD_FILESYSTEM}
    train_unet_darknet2dl)

# Tanks windows for this unneeded workaround
if(MSVC)
    if("${CUSTOM_TORCH_BUILD_PATH}" STREQUAL "")
        find_package(Torch REQUIRED)
    else()
        find_package(Torch REQUIRED PATHS ${CUSTOM_TORCH_BUILD_PATH} NO_DEFAULT_PATH)
    endif()

    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    foreach(TORCH_DLL ${TORCH_DLLS})
        message(STATUS TORCH_DLL=${TORCH_DLL})
        file(COPY ${TORCH_DLL} DESTINATION ${CMAKE_BINARY_DIR})
    endforeach()
    file(GLOB OPENCV_DLLS "${CUSTOM_OPENCV_BUILD_PATH}/x64/vc15/bin/*.dll")
    foreach(OPENCV_DLL ${OPENCV_DLLS})
        message(STATUS OPENCV_DLL=${OPENCV_DLL})
        file(COPY ${OPENCV_DLL} DESTINATION ${CMAKE_BINARY_DIR})
    endforeach()
    set(QTPLATFORM_FOLDER "C:/Qt/5.15.2/msvc2015_64/plugins/platforms")
    file(COPY ${QTPLATFORM_FOLDER} DESTINATION ${CMAKE_BINARY_DIR})
    #file(COPY "C:/Qt/5.15.2/msvc2015_64/bin/Qt5Cored.dll" DESTINATION ${CMAKE_BINARY_DIR})
    #file(COPY "C:/Qt/5.15.2/msvc2015_64/bin/Qt5Guid.dll" DESTINATION ${CMAKE_BINARY_DIR})
    #file(COPY "C:/Qt/5.15.2/msvc2015_64/bin/Qt5Widgetsd.dll" DESTINATION ${CMAKE_BINARY_DIR})
endif(MSVC)

#qt5_create_translation(QM_FILES ${CMAKE_SOURCE_DIR} ${TS_FILES})