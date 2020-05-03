SET(NNPACK_SEARCH_PATHS
        ${NNPACK_ROOT}
        $ENV{NNPACK_ROOT}
        /opt/NNPACK
        /usr/local
        /usr
        )

FIND_PATH(NNPACK_INCLUDE_DIR NAMES nnpack.h PATHS ${NNPACK_SEARCH_PATHS} PATH_SUFFIXES include)
FIND_PATH(PTHREADPOOL_INCLUDE_DIR NAMES pthreadpool.h PATHS ${NNPACK_SEARCH_PATHS} PATH_SUFFIXES deps/pthreadpool/include)
FIND_LIBRARY(NNPACK_LIBRARIES NAMES nnpack PATHS ${NNPACK_SEARCH_PATHS} PATH_SUFFIXES lib lib64)
FIND_LIBRARY(PTHREADPOOL_LIBRARIES NAMES pthreadpool PATHS ${NNPACK_SEARCH_PATHS} PATH_SUFFIXES lib lib64)

SET(NNPACK_FOUND ON)

#    Check include files
IF (NOT NNPACK_INCLUDE_DIR)
    SET(NNPACK_FOUND OFF)
    MESSAGE(STATUS "Could not find NNPACK include. Turning NNPACK_FOUND off")
ENDIF ()

IF (NOT PTHREADPOOL_INCLUDE_DIR)
    SET(NNPACK_FOUND OFF)
    MESSAGE(STATUS "Could not find pthreadpool include. Turning NNPACK_FOUND off")
ENDIF ()

#    Check libraries
IF (NOT NNPACK_LIBRARIES)
    SET(NNPACK_FOUND OFF)
    MESSAGE(STATUS "Could not find NNPACK lib. Turning NNPACK_FOUND off")
ENDIF ()

IF (NOT PTHREADPOOL_LIBRARIES)
    SET(NNPACK_FOUND OFF)
    MESSAGE(STATUS "Could not find pthreadpool lib. Turning NNPACK_FOUND off")
ENDIF ()

IF (NNPACK_FOUND)
    add_definitions(-DUSE_NNPACK)
    IF (NOT NNPACK_FIND_QUIETLY)
        MESSAGE(STATUS "Found NNPACK (include: ${NNPACK_INCLUDE_DIR}, library: ${NNPACK_LIBRARIES})")
        MESSAGE(STATUS "Found pthreadpool (include: ${PTHREADPOOL_INCLUDE_DIR}, library: ${PTHREADPOOL_LIBRARIES})")
    ENDIF (NOT NNPACK_FIND_QUIETLY)
ELSE (NNPACK_FOUND)
    IF (NNPACK_FIND_REQUIRED)
        MESSAGE(FATAL_ERROR "Could not find NNPACK")
    ENDIF (NNPACK_FIND_REQUIRED)
ENDIF (NNPACK_FOUND)

MARK_AS_ADVANCED(
        NNPACK_INCLUDE_DIR
        NNPACK_LIB
        NNPACK
)