cmake_minimum_required(VERSION 3.15)
project(EigenDownloadTest)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(
    eigen_test
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    TIMEOUT 30  # 30 second timeout
)

# Try to download, but don't fail the build if it doesn't work
FetchContent_MakeAvailable(eigen_test)

# If we get here without error, the test passed
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/eigen_download_success.txt "Download succeeded")
