#ifndef ALLO_TD
#define ALLO_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// allo dialect definition.
//===----------------------------------------------------------------------===//

def Allo_Dialect : Dialect {
    let name = "allo";
    let summary = "Allo Dialect";
    let description = [{
        Allo is a dialect designed for accelerator programming and designing
    }];
    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;
    let cppNamespace = "::mlir::allo";
}

//===----------------------------------------------------------------------===//
// Base allo definitions.
//===----------------------------------------------------------------------===//

class Allo_Op<string mnemonic, list<Trait> traits = []> :
        Op<Allo_Dialect, mnemonic, traits>;

class Allo_Type<string mnemonic, list<Trait> traits = []> :
        TypeDef<Allo_Dialect, mnemonic, traits>;

class Allo_Attr<string mnemonic, list<Trait> traits = []> :
        AttrDef<Allo_Dialect, mnemonic, traits>;

#endif // ALLO_TD
