# These lists are later turned into target properties on main caffe library target
set(Caffe_LINKER_LIBS "")
set(Caffe_INCLUDE_DIRS "")
set(Caffe_DEFINITIONS "")
set(Caffe_COMPILE_OPTIONS "")

if (EXISTS ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup()
    list(APPEND Caffe_LINKER_LIBS PUBLIC ${CONAN_LIBS})
    list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${CONAN_INCLUDE_DIRS})
    list(APPEND Caffe_DEFINITIONS PUBLIC ${CONAN_DEFINES})
    if (DEFINED CONAN_XTENSOR_ROOT)
        list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_XTENSOR)
    endif ()
else ()
    message(WARNING "The file conanbuildinfo.cmake doesn't exist, you have to run conan install first")
endif ()

# ---[ Boost
if (NOT DEFINED CONAN_BOOST_ROOT)
    find_package(Boost 1.54 REQUIRED COMPONENTS system thread filesystem)
    list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${Boost_INCLUDE_DIRS})
    list(APPEND Caffe_LINKER_LIBS PUBLIC ${Boost_LIBRARIES})
endif ()
list(APPEND Caffe_DEFINITIONS PUBLIC -DBOOST_ALL_NO_LIB)

if (DEFINED MSVC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18.0.40629.0)
    # Required for VS 2013 Update 4 or earlier.
    list(APPEND Caffe_DEFINITIONS PUBLIC -DBOOST_NO_CXX11_TEMPLATE_ALIASES)
endif ()

# ---[ Threads
find_package(Threads REQUIRED)
list(APPEND Caffe_LINKER_LIBS PRIVATE ${CMAKE_THREAD_LIBS_INIT})

# ---[ OpenMP
if (USE_OPENMP)
    # Ideally, this should be provided by the BLAS library IMPORTED target. However,
    # nobody does this, so we need to link to OpenMP explicitly and have the maintainer
    # to flick the switch manually as needed.
    #
    # Moreover, OpenMP package does not provide an IMPORTED target as well, and the
    # suggested way of linking to OpenMP is to append to CMAKE_{C,CXX}_FLAGS.
    # However, this naïve method will force any user of Caffe to add the same kludge
    # into their buildsystem again, so we put these options into per-target PUBLIC
    # compile options and link flags, so that they will be exported properly.
    if (APPLE)
        if (CMAKE_C_COMPILER_ID MATCHES "Clang")
            set(OpenMP_C "${CMAKE_C_COMPILER}")
            set(OpenMP_C_FLAGS "-fopenmp=libomp" "-fopenmp")
            set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
            set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
            set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
            set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
        endif ()
        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
            set(OpenMP_CXX_FLAGS "-fopenmp=libomp" "-fopenmp")
            set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
            set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
            set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
            set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        endif ()
        link_directories("${CMAKE_INSTALL_PREFIX}/lib")
    endif ()
    find_package(OpenMP REQUIRED)
    list(APPEND Caffe_LINKER_LIBS PRIVATE ${OpenMP_CXX_FLAGS})
    list(APPEND Caffe_COMPILE_OPTIONS PRIVATE ${OpenMP_CXX_FLAGS})
    add_definitions(-DUSE_OMP)
endif ()

# ---[ Google-glog
if (NOT DEFINED CONAN_GLOG_ROOT)
    include("cmake/External/glog.cmake")
    list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${GLOG_INCLUDE_DIRS})
    list(APPEND Caffe_LINKER_LIBS PUBLIC ${GLOG_LIBRARIES})
endif ()

# ---[ Google-gflags
if (NOT DEFINED CONAN_GFLAGS_ROOT)
    include("cmake/External/gflags.cmake")
    list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${GFLAGS_INCLUDE_DIRS})
    list(APPEND Caffe_LINKER_LIBS PUBLIC ${GFLAGS_LIBRARIES})
endif ()

# ---[ Google-protobuf
include(cmake/ProtoBuf.cmake)

# ---[ HDF5
if (NOT DEFINED CONAN_HDF5_ROOT)
    if (MSVC)
        # Find HDF5 using it's hdf5-config.cmake file with MSVC
        if (DEFINED HDF5_DIR)
            list(APPEND CMAKE_MODULE_PATH ${HDF5_DIR})
        endif ()
        if (BUILD_SHARED_LIBS)
            set(LIB_TYPE SHARED) # or SHARED
        else ()
            set(LIB_TYPE STATIC)
        endif ()
        string(TOLOWER ${LIB_TYPE} SEARCH_TYPE)
        find_package(HDF5 NAMES hdf5 COMPONENTS C HL ${SEARCH_TYPE})
        set(HDF5_LIBRARIES ${HDF5_C_${LIB_TYPE}_LIBRARY})
        set(HDF5_HL_LIBRARIES ${HDF5_HL_${LIB_TYPE}_LIBRARY})
        if (NOT HDF5_INCLUDE_DIRS)
            set(HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIR})
        endif ()
    else ()
        find_package(HDF5 COMPONENTS HL REQUIRED)
    endif ()
    list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${HDF5_INCLUDE_DIRS})
    list(APPEND Caffe_LINKER_LIBS PUBLIC ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
endif ()
add_definitions(-DUSE_HDF5)

# ---[ LMDB
if (NOT DEFINED CONAN_LMDB_ROOT)
    find_package(LMDB REQUIRED)
    list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${LMDB_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS PUBLIC ${LMDB_LIBRARIES})
endif ()
list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_LMDB)
if (ALLOW_LMDB_NOLOCK)
    list(APPEND Caffe_DEFINITIONS PRIVATE -DALLOW_LMDB_NOLOCK)
endif ()

# ---[ LevelDB
if (USE_LEVELDB)
    find_package(LevelDB REQUIRED)
    list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${LevelDB_INCLUDES})
    list(APPEND Caffe_LINKER_LIBS PUBLIC ${LevelDB_LIBRARIES})
    list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_LEVELDB)
endif ()

# ---[ Snappy
if (USE_LEVELDB)
    find_package(Snappy REQUIRED)
    list(APPEND Caffe_INCLUDE_DIRS PRIVATE ${Snappy_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS PRIVATE ${Snappy_LIBRARIES})
endif ()

# ---[ CUDA
include(cmake/Cuda.cmake)
if (NOT HAVE_CUDA)
    if (CPU_ONLY)
        message(STATUS "-- CUDA is disabled. Building without it...")
    else ()
        message(WARNING "-- CUDA is not detected by cmake. Building without it...")
    endif ()

    list(APPEND Caffe_DEFINITIONS PUBLIC -DCPU_ONLY)
endif ()

if (USE_NCCL)
    include("cmake/External/nccl.cmake")
    include_directories(SYSTEM ${NCCL_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS ${NCCL_LIBRARIES})
    add_definitions(-DUSE_NCCL)
endif ()

if (USE_NNPACK)
    find_package(NNPACK REQUIRED)
    include_directories(SYSTEM ${NNPACK_INCLUDE_DIR} ${PTHREADPOOL_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS ${NNPACK_LIBRARIES} ${PTHREADPOOL_LIBRARIES})
    if (NNPACK_FOUND)
        set(HAVE_NNPACK TRUE)
    else ()
        set(HAVE_NNPACK FALSE)
    endif ()
endif ()

# ---[ OpenCV
if (NOT DEFINED CONAN_OPENCV_ROOT)
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs videoio video)
    if (NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
        find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc videoio)
    endif ()
    list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${OpenCV_INCLUDE_DIRS})
    list(APPEND Caffe_LINKER_LIBS PUBLIC ${OpenCV_LIBS})
    message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
endif ()
list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_OPENCV)

# ---[ BLAS
if (NOT APPLE)
    set(BLAS "Atlas" CACHE STRING "Selected BLAS library")
    set_property(CACHE BLAS PROPERTY STRINGS "Atlas;Open;MKL")

    if (BLAS STREQUAL "Atlas" OR BLAS STREQUAL "atlas")
        find_package(Atlas REQUIRED)
        list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${Atlas_INCLUDE_DIR})
        list(APPEND Caffe_LINKER_LIBS PUBLIC ${Atlas_LIBRARIES})
    elseif (BLAS STREQUAL "Open" OR BLAS STREQUAL "open")
        find_package(OpenBLAS REQUIRED)
        list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${OpenBLAS_INCLUDE_DIR})
        list(APPEND Caffe_LINKER_LIBS PUBLIC ${OpenBLAS_LIB})
    elseif (BLAS STREQUAL "MKL" OR BLAS STREQUAL "mkl")
        find_package(MKL REQUIRED)
        list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${MKL_INCLUDE_DIR})
        list(APPEND Caffe_LINKER_LIBS PUBLIC ${MKL_LIBRARIES})
        list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_MKL)
    endif ()
elseif (APPLE)
    find_package(vecLib REQUIRED)
    list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${vecLib_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS PUBLIC ${vecLib_LINKER_LIBS})

    if (VECLIB_FOUND)
        if (NOT vecLib_INCLUDE_DIR MATCHES "^/System/Library/Frameworks/vecLib.framework.*")
            list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_ACCELERATE)
        endif ()
    endif ()
endif ()

# ---[ Python
if (BUILD_python)
    if (NOT "${python_version}" VERSION_LESS "3.0.0")
        # use python3
        find_package(PythonInterp 3.0)
        find_package(PythonLibs 3.0)
        find_package(NumPy 1.7.1)
        # Find the matching boost python implementation
        set(version ${PYTHONLIBS_VERSION_STRING})
        STRING(REGEX REPLACE "[^0-9]" "" boost_py_version ${version})
        find_package(Boost 1.46 COMPONENTS "python${boost_py_version}")
        set(Boost_PYTHON_FOUND ${Boost_PYTHON${boost_py_version}_FOUND})

        while (NOT "${version}" STREQUAL "" AND NOT Boost_PYTHON_FOUND)
            STRING(REGEX REPLACE "([0-9.]+).[0-9]+" "\\1" version ${version})

            STRING(REGEX REPLACE "[^0-9]" "" boost_py_version ${version})
            find_package(Boost 1.46 COMPONENTS "python${boost_py_version}")
            set(Boost_PYTHON_FOUND ${Boost_PYTHON${boost_py_version}_FOUND})

            STRING(REGEX MATCHALL "([0-9.]+).[0-9]+" has_more_version ${version})
            if ("${has_more_version}" STREQUAL "" OR Boost_PYTHON_FOUND)
                break()
            endif ()
        endwhile ()
        if (NOT Boost_PYTHON_FOUND)
            find_package(Boost 1.46 COMPONENTS python)
        endif ()
    else ()
        # disable Python 3 search
        find_package(PythonInterp 2.7)
        find_package(PythonLibs 2.7)
        find_package(NumPy 1.7.1)
        find_package(Boost 1.46 COMPONENTS python)
    endif ()
    if (PYTHONLIBS_FOUND AND NUMPY_FOUND AND Boost_PYTHON_FOUND)
        set(HAVE_PYTHON TRUE)
        if (Boost_USE_STATIC_LIBS AND MSVC)
            list(APPEND Caffe_DEFINITIONS PUBLIC -DBOOST_PYTHON_STATIC_LIB)
        endif ()
        if (BUILD_python_layer)
            list(APPEND Caffe_DEFINITIONS PRIVATE -DWITH_PYTHON_LAYER)
            list(APPEND Caffe_INCLUDE_DIRS PRIVATE ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR} PUBLIC ${Boost_INCLUDE_DIRS})
            list(APPEND Caffe_LINKER_LIBS PRIVATE ${PYTHON_LIBRARIES} PUBLIC ${Boost_LIBRARIES})
        endif ()
    endif ()
endif ()

# ---[ Matlab
if (BUILD_matlab)
    if (MSVC)
        find_package(Matlab COMPONENTS MAIN_PROGRAM MX_LIBRARY)
        if (MATLAB_FOUND)
            set(HAVE_MATLAB TRUE)
        endif ()
    else ()
        find_package(MatlabMex)
        if (MATLABMEX_FOUND)
            set(HAVE_MATLAB TRUE)
        endif ()
    endif ()
    # sudo apt-get install liboctave-dev
    find_program(Octave_compiler NAMES mkoctfile DOC "Octave C++ compiler")

    if (HAVE_MATLAB AND Octave_compiler)
        set(Matlab_build_mex_using "Matlab" CACHE STRING "Select Matlab or Octave if both detected")
        set_property(CACHE Matlab_build_mex_using PROPERTY STRINGS "Matlab;Octave")
    endif ()
endif ()

# ---[ Doxygen
if (BUILD_docs)
    find_package(Doxygen)
endif ()
