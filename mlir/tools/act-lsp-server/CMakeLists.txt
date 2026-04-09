get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_tool(act-lsp-server act-lsp-server.cpp)

llvm_update_compile_flags(act-lsp-server)

add_dependencies(act-lsp-server MLIRActIncGen)

target_link_libraries(act-lsp-server
    PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    MLIRIR
    MLIRLspServerLib
    MLIRAct
    MLIRRegisterAllDialects
    MLIRRegisterAllExtensions
)

