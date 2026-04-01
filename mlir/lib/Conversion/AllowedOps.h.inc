#ifdef GET_LINALG_OPS
#undef GET_LINALG_OPS
linalg::GenericOp,
linalg::MapOp,
linalg::ReduceOp,
linalg::FillOp,
linalg::ContractOp,
linalg::PackOp,
linalg::UnPackOp,
linalg::YieldOp,
linalg::MatmulOp,
linalg::SoftmaxOp
#endif

#ifdef GET_TENSOR_OPS
#undef GET_TENSOR_OPS
tensor::CastOp,
tensor::SplatOp,
tensor::PadOp,
tensor::ConcatOp,
tensor::BitcastOp,
tensor::ExpandShapeOp,
tensor::CollapseShapeOp,
tensor::ExtractSliceOp,
tensor::InsertSliceOp,
tensor::EmptyOp
#endif
