add_mlir_library(MLIRAlloTranslation
        BaseEmitter.cpp
        BaseHLSEmitter.cpp

        LINK_LIBS PUBLIC
        MLIRAnalysis
        MLIRAllo
)