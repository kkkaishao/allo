#ifndef ACT_TD
#define ACT_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Act_Dialect : Dialect {
  let name = "act";
  let summary = "ACT dialect for describing tensor accelerator ISAs";
  let description = [{
    The ACT dialect captures hardware ISA descriptions for tensor accelerators.
    It models hardware resources (buffers, descriptors, states), memory access
    patterns (strided, tiled), and instruction semantics (define/emit) to enable
    automatic compiler backend generation from ISA specifications.
  }];
  let useDefaultTypePrinterParser = 1;
  // let useDefaultAttributePrinterParser = 1;
  let cppNamespace = "::mlir::act";
}

class Act_Op<string mnemonic, list<Trait> traits = []> :
        Op<Act_Dialect, mnemonic, traits>;

class Act_Type<string mnemonic, list<Trait> traits = []> :
        TypeDef<Act_Dialect, mnemonic, traits>;

class Act_Attr<string mnemonic, list<Trait> traits = []> :
        AttrDef<Act_Dialect, mnemonic, traits>;

#endif // ACT_TD
