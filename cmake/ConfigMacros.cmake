MACRO(REQUIRE_HEADER header)
  FIND_PATH(header_path ${header})
  IF(${header_path} STREQUAL "")
    MESSAGE(SEND_ERROR "Required header ${header} not found")
  ELSE(${header_path} STREQUAL "")
    INCLUDE_DIRECTORIES(${header_path})
  ENDIF(${header_path} STREQUAL "")
ENDMACRO(REQUIRE_HEADER header)

MACRO(FIND_AND_LINK_LIBRARY library target)
  FIND_LIBRARY(${library}_LIBRARY ${library})
  IF(${${library}_LIBRARY} STREQUAL ${library}_LIBRARY-NOTFOUND)
    MESSAGE(SEND_ERROR "Required library ${library} not found")
  ELSE(${${library}_LIBRARY} STREQUAL ${library}_LIBRARY-NOTFOUND)
    TARGET_LINK_LIBRARIES(${target} ${${library}_LIBRARY})
  ENDIF(${${library}_LIBRARY} STREQUAL ${library}_LIBRARY-NOTFOUND)
ENDMACRO(FIND_AND_LINK_LIBRARY library target)
