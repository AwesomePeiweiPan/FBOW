PROJECT(fbow)

IF(NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE "Debug" )
ENDIF()

find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS}) 
     
set(FBOW_INCLUDE_DIRS "/usr/local/include")
set(FBOW_LIBS "/usr/local/lib/libfbow.so")

# 这一行是关键，确保加入此行
INCLUDE_DIRECTORIES(${FBOW_INCLUDE_DIRS})

ADD_EXECUTABLE(image_matching image_matching.cpp)
target_link_libraries(image_matching ${OpenCV_LIBS} ${FBOW_LIBS})
