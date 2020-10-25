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
find_package(OpenCV REQUIRED)

set(TS_FILES UnetTrainingTool_ru_RU.ts)

include_directories(
        #third_party/???/include
        ${OpenCV_INCLUDE_DIRS})

if(ANDROID)
    add_library(${PROJECT_NAME} SHARED
            main.cpp
            OpenDatasetsDialog.cpp
            OpenDatasetsDialog.hpp
            #${TS_FILES}
            )
else()
    add_executable(${PROJECT_NAME}
            main.cpp
            OpenDatasetsDialog.cpp
            OpenDatasetsDialog.hpp
            #${TS_FILES}
            )
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE
        Qt${QT_VERSION_MAJOR}::Widgets
        ${OpenCV_LIBS})

#qt5_create_translation(QM_FILES ${CMAKE_SOURCE_DIR} ${TS_FILES})