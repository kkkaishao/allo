nanobind_add_module(_liballo
    NB_STATIC STABLE_ABI
    ir.cpp
    ops.cpp
    main.cpp
    allo.cpp
    transform.cpp
)

# turn off -Wcast-qual for nanobind
target_compile_options(nanobind-static 
    PUBLIC 
    "-Wno-cast-qual" 
    "-Wno-covered-switch-default"
)

set_target_properties(_liballo PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/allo/experimental/_C/liballo"
    CXX_VISIBILITY_PRESET hidden
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(_liballo
    PRIVATE
    # MLIR Libraries
    ${dialect_libs}
    ${conversion_libs}
    
    MLIRAllo
    MLIRAlloTransform
    MLIRAlloTransformOps

    Python3::Module
)

if (WIN32)
    target_link_libraries(_liballo PRIVATE ${CMAKE_DL_LIBS})
    set_target_properties(_liballo PROPERTIES SUFFIX ".pyd")
    set_target_properties(_liballo PROPERTIES PREFIX "lib")
elseif (DARWIN)
    target_link_options(_liballo PRIVATE "-Wl,-no_warn_duplicate_libraries")
    target_link_libraries(_liballo PRIVATE z)
else ()
    target_link_libraries(_liballo PRIVATE z)
endif ()
target_link_options(_liballo PRIVATE ${LLVM_LDFLAGS})
