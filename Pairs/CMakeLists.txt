PROJECT(fbow)

set (CMAKE_CXX_FLAGS "-lstdc++fs -std=c++17")
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

IF(NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE "Debug" )
ENDIF()

find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS}) 

find_package(Threads REQUIRED)
     
set(FBOW_INCLUDE_DIRS "/usr/local/include")
set(FBOW_LIBS "/usr/local/lib/libfbow.so")

# 这一行是关键，确保加入此行
INCLUDE_DIRECTORIES(${FBOW_INCLUDE_DIRS})

ADD_EXECUTABLE(Group_Sequence_Set Group_Sequence_Set.cpp)
target_link_libraries(Group_Sequence_Set ${OpenCV_LIBS} ${FBOW_LIBS} Threads::Threads)

ADD_EXECUTABLE(Loop_Created Loop_Created.cpp)
target_link_libraries(Loop_Created ${OpenCV_LIBS} ${FBOW_LIBS} Threads::Threads)

ADD_EXECUTABLE(New_Loop_Created New_Loop.cpp)
target_link_libraries(New_Loop_Created ${OpenCV_LIBS} ${FBOW_LIBS} Threads::Threads)

