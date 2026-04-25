#cmake_minimum_required(VERSION 3.10)
#
#project(FalezCSNNSimulator)
#
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto") # for turbo on grid5000
#
#
#option(USE_GUI "Enable GUI" OFF)
## option(USE_GUI "Enable GUI" ON)
#
#if(USE_GUI)
#    message(STATUS "GUI Enable")
#else()
#    message(STATUS "GUI Disable")
#endif()
#
#
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)
#
#set(CMAKE_CXX_STANDARD 17)
#set(LIBRARY_NAME "CSNNS" CACHE STRING "Name of library")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")
#
## set(CMAKE_AUTOMOC ON)
#set(CMAKE_INCLUDE_CURRENT_DIR ON)
#
#if(USE_GUI)
#    set(CMAKE_AUTOMOC ON)
#    find_package(Qt4 4.8.7 REQUIRED QtCore QtGui)
#    add_definitions(-DENABLE_QT)
#
#    include_directories(include/ ${QT_INCLUDES})
#endif()
#
#find_package(OpenCV REQUIRED)
#find_package(Threads REQUIRED)
#
#find_package(BLAS REQUIRED)
#find_package(LAPACK REQUIRED)
#
#message(STATUS ${LAPACK_INCLUDES})
#
#list(FILTER SOURCES EXCLUDE REGEX "apps/benchmarks/Bench7\\.cpp")
#list(FILTER SOURCES EXCLUDE REGEX "apps/benchmarks/BenchCifar10WP\\.cpp")
#list(FILTER SOURCES EXCLUDE REGEX "apps/cifar/Cifar10_L1\\.cpp")
#list(FILTER SOURCES EXCLUDE REGEX "apps/cifar/Cifar10_L2\\.cpp")
#list(FILTER SOURCES EXCLUDE REGEX "apps/whitening/GenerateCifar10Whiten\\.cpp")
#list(FILTER SOURCES EXCLUDE REGEX "apps/whitening/GenerateSTL10Whiten\\.cpp")
#
#
#file(GLOB_RECURSE SOURCES src/* include/*)
##include_directories(include/ ${LAPACK_INCLUDES})
#include_directories(${CMAKE_SOURCE_DIR} include/ dep/ ${LAPACK_INCLUDES})
#include_directories( ${OpenCV_INCLUDE_DIRS} )
#
#
#set(APPS_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -ansi -pedantic -Wshadow -Weffc++ -ftemplate-backtrace-limit=0 -ltbb2 -msse -msse2 -msse3 -march=native -mfpmath=sse")
## add_definitions(-DSMID_AVX256)
#
#add_subdirectory(dep/libsvm)
#
#if(USE_GUI)
#    add_subdirectory(dep/qcustomplot)
#endif()
#
#
##
##   Build library
##
#
#add_library(${LIBRARY_NAME} SHARED ${SOURCES})
#set_target_properties(${LIBRARY_NAME} PROPERTIES COMPILE_FLAGS ${APPS_FLAGS})
#target_link_libraries(${LIBRARY_NAME} ${BLAS_LIBRARIES} ${LAPACK_lapack_LIBRARY} libsvm Threads::Threads ${OpenCV_LIBS} tbb)
#if(USE_GUI)
#    target_link_libraries(${LIBRARY_NAME} Qt4::QtCore Qt4::QtGui qcustomplot)
#endif()
#
##   Build executables
##
#file(GLOB_RECURSE apps apps/*)
#foreach(app ${apps})
#    get_filename_component(app_name ${app} NAME_WE)
#    add_executable(${app_name} ${app})
#    set_target_properties(${app_name} PROPERTIES COMPILE_FLAGS ${APPS_FLAGS})
#    target_link_libraries(${app_name} ${LIBRARY_NAME} ${BLAS_LIBRARIES} ${LAPACK_lapack_LIBRARY} Threads::Threads ${OpenCV_LIBS} tbb)
#    if(USE_GUI)
#        target_link_libraries(${app_name} ${LIBRARY_NAME} Qt4::QtCore Qt4::QtGui)
#    endif()
#endforeach()

# ============================================================================
# CMakeLists.txt — modificări propuse pentru paralelism real
# ============================================================================
# Aceste schimbări NU sunt aplicate. Le verifici tu, apoi le copiezi peste
# CMakeLists.txt dacă ești ok.
#
# Probleme cu CMakeLists.txt actual:
#
#   1. CMAKE_BUILD_TYPE nu e setat -> build poate fi Debug -> -O0 -g, asserts
#      active. Toate flag-urile -O3/-march=native pe care le-ai adăugat sunt
#      anulate dacă alegi Debug.
#
#   2. -flto e doar în CMAKE_CXX_FLAGS dar NU și în linker flags. Pe gcc, LTO
#      trebuie pasat și la link, altfel obiectele LTO nu sunt optimizate
#      inter-procedural. Soluția curată: CMAKE_INTERPROCEDURAL_OPTIMIZATION.
#
#   3. APPS_FLAGS conține "-ltbb2" care e flag de LINKER, nu de COMPILE -> e
#      ignorat tăcut de compilator. TBB e linkat doar pentru că ai "tbb" în
#      target_link_libraries (linia 74), dar -ltbb2 e literalmente inutil.
#
#   4. -msse -msse2 -msse3 -mfpmath=sse — toate suprascrise/făcute redundante
#      de -march=native (care pe Xeon Gold 6254 activează AVX-512). Le scoatem.
#
#   5. -Weffc++ -pedantic -Wshadow — generează tone de warnings irelevante,
#      încetinesc compilarea. Le scot din build-ul de release.
#
#   6. Nu există find_package(TBB) explicit -> dacă std::execution::par e
#      folosit (cum face acum ConvolutionSampler3D.cpp), gcc cere link cu TBB.
#      Adăugăm find_package(TBB) și folosim TBB::tbb la link.
#
# ============================================================================

cmake_minimum_required(VERSION 3.12)  # 3.12 pentru CMAKE_INTERPROCEDURAL_OPTIMIZATION

project(FalezCSNNSimulator)

# --- (1) Build type default = Release ---
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Optimizări globale, doar pe Release ---
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG") #for roazhon-13
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=x86-64-v3 -DNDEBUG")

# --- (2) LTO global (gestionează corect compile + link) ---
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_ok OUTPUT ipo_err)
if(ipo_ok)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "IPO/LTO enabled")
else()
    message(STATUS "IPO/LTO not supported: ${ipo_err}")
endif()

option(USE_GUI "Enable GUI" OFF)

if(USE_GUI)
    message(STATUS "GUI Enable")
else()
    message(STATUS "GUI Disable")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(LIBRARY_NAME "CSNNS" CACHE STRING "Name of library")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(USE_GUI)
    set(CMAKE_AUTOMOC ON)
    find_package(Qt4 4.8.7 REQUIRED QtCore QtGui)
    add_definitions(-DENABLE_QT)
    include_directories(include/ ${QT_INCLUDES})
endif()

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# --- (6) TBB explicit pentru std::execution::par ---
find_package(TBB REQUIRED)
message(STATUS "TBB found: ${TBB_VERSION}")

message(STATUS ${LAPACK_INCLUDES})

list(FILTER SOURCES EXCLUDE REGEX "apps/benchmarks/Bench7\\.cpp")
list(FILTER SOURCES EXCLUDE REGEX "apps/benchmarks/BenchCifar10WP\\.cpp")
list(FILTER SOURCES EXCLUDE REGEX "apps/cifar/Cifar10_L1\\.cpp")
list(FILTER SOURCES EXCLUDE REGEX "apps/cifar/Cifar10_L2\\.cpp")
list(FILTER SOURCES EXCLUDE REGEX "apps/whitening/GenerateCifar10Whiten\\.cpp")
list(FILTER SOURCES EXCLUDE REGEX "apps/whitening/GenerateSTL10Whiten\\.cpp")

file(GLOB_RECURSE SOURCES src/* include/*)
include_directories(${CMAKE_SOURCE_DIR} include/ dep/ ${LAPACK_INCLUDES})
include_directories(${OpenCV_INCLUDE_DIRS})

# --- (3,4,5) APPS_FLAGS curățate: scoatem -ltbb2, -msse*, -mfpmath=sse,
#     -Weffc++, -pedantic. Păstrăm doar -Wall pentru warnings utile.
#     -O3 -march=native vin acum din CMAKE_CXX_FLAGS_RELEASE, nu le mai
#     dublăm aici.
set(APPS_FLAGS "-Wall -Wextra")

add_subdirectory(dep/libsvm)

if(USE_GUI)
    add_subdirectory(dep/qcustomplot)
endif()

# ----------------------------------------------------------------------------
# Library
# ----------------------------------------------------------------------------
add_library(${LIBRARY_NAME} SHARED ${SOURCES})
set_target_properties(${LIBRARY_NAME} PROPERTIES COMPILE_FLAGS ${APPS_FLAGS})
target_link_libraries(${LIBRARY_NAME}
        ${BLAS_LIBRARIES}
        ${LAPACK_lapack_LIBRARY}
        libsvm
        Threads::Threads
        ${OpenCV_LIBS}
        TBB::tbb            # <-- target modern, nu "tbb" string
)
if(USE_GUI)
    target_link_libraries(${LIBRARY_NAME} Qt4::QtCore Qt4::QtGui qcustomplot)
endif()

# ----------------------------------------------------------------------------
# Executables
# ----------------------------------------------------------------------------
file(GLOB_RECURSE apps apps/*)
foreach(app ${apps})
    get_filename_component(app_name ${app} NAME_WE)
    add_executable(${app_name} ${app})
    set_target_properties(${app_name} PROPERTIES COMPILE_FLAGS ${APPS_FLAGS})
    target_link_libraries(${app_name}
            ${LIBRARY_NAME}
            ${BLAS_LIBRARIES}
            ${LAPACK_lapack_LIBRARY}
            Threads::Threads
            ${OpenCV_LIBS}
            TBB::tbb
    )
    if(USE_GUI)
        target_link_libraries(${app_name} ${LIBRARY_NAME} Qt4::QtCore Qt4::QtGui)
    endif()
endforeach()

# ============================================================================
# Cum compilezi după ce înlocuiești CMakeLists.txt:
#
#   rm -rf csnn-simulator-build
#   mkdir csnn-simulator-build && cd csnn-simulator-build
#   cmake -DCMAKE_BUILD_TYPE=Release ..
#   make -j$(nproc)
#
# Verifici la output că vezi:
#   -- IPO/LTO enabled
#   -- TBB found: <version>
#
# Dacă find_package(TBB) eșuează pe roazhon13:
#   module load tbb        # sau ce module manager folosesc
#   sau:
#   apt-get install libtbb-dev   (dacă ai sudo)
#
# La rulare, ca să folosești NUMA-aware allocation pe nodul cu 2 sockets:
#   numactl --interleave=all ./KTH_3D
# ============================================================================
