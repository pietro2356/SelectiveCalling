find_package(PkgConfig)

PKG_CHECK_MODULES(PC_GR_SELCAL gnuradio-selcal)

FIND_PATH(
    GR_SELCAL_INCLUDE_DIRS
    NAMES gnuradio/selcal/api.h
    HINTS $ENV{SELCAL_DIR}/include
        ${PC_SELCAL_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    GR_SELCAL_LIBRARIES
    NAMES gnuradio-selcal
    HINTS $ENV{SELCAL_DIR}/lib
        ${PC_SELCAL_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/gnuradio-selcalTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GR_SELCAL DEFAULT_MSG GR_SELCAL_LIBRARIES GR_SELCAL_INCLUDE_DIRS)
MARK_AS_ADVANCED(GR_SELCAL_LIBRARIES GR_SELCAL_INCLUDE_DIRS)
