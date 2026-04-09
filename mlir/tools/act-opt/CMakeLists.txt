get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_tool(act-opt act-opt.cpp)

llvm_update_compile_flags(act-opt)

add_dependencies(act-opt MLIRActIncGen)

target_link_libraries(act-opt
    PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    MLIRIR
    MLIROptLib
    MLIRAct
    MLIRActConversion
    MLIRRegisterAllDialects
    MLIRRegisterAllExtensions
    MLIRRegisterAllPasses
)

