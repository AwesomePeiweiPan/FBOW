PROJECT(VocCreate)
set (CMAKE_CXX_FLAGS "-lstdc++fs -std=c++17")
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

IF(NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE "Debug" )
ENDIF()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")

find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS}) 
INCLUDE_DIRECTORIES(/home/peiweipan/fbow/src)

# 添加 fbow 相关的源文件
set(FBOW_SOURCES /home/peiweipan/fbow/src/fbow.cpp /home/peiweipan/fbow/src/vocabulary_creator.cpp)

ADD_EXECUTABLE(fbow_create_voc_step0 fbow_create_voc_step0.cpp ${FBOW_SOURCES})
ADD_EXECUTABLE(fbow_create_voc_step1 fbow_create_voc_step1.cpp ${FBOW_SOURCES})

target_link_libraries(fbow_create_voc_step0 ${OpenCV_LIBS})
target_link_libraries(fbow_create_voc_step1 ${OpenCV_LIBS})

target_link_libraries(fbow_create_voc_step0 ${OpenCV_LIBS} pthread)
target_link_libraries(fbow_create_voc_step1 ${OpenCV_LIBS} pthread)

