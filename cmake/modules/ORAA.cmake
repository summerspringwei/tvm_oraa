if(USE_ORAA)
  message(STATUS "Build with ORAA support")
  list(APPEND COMPILER_SRCS src/target/opt/build_oraa_on.cc)
  list(APPEND COMPILER_SRCS src/runtime/oraa/oraa_module.cc)
endif(USE_ORAA)