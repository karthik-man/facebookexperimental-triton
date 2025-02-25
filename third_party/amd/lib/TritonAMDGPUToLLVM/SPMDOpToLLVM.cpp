#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

using namespace mlir;

namespace {

struct GetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern<
      triton::GetNumProgramsOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                    mlir::gpu::Dimension::y,
                                                    mlir::gpu::Dimension::z};
    Location loc = op->getLoc();
    assert(op.getAxisAsInt() < 3);
    Value blockId =
        rewriter.create<::mlir::gpu::GridDimOp>(loc, dims[op.getAxisAsInt()]);
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, i32_ty, blockId);
    return success();
  }
};

struct ArriveBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::ArriveBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::ArriveBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Block *currentBlock = rewriter.getInsertionBlock();
    // Block *afterCondBarBlock =
    //     rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    // Block *trueBlock = rewriter.createBlock(afterCondBarBlock);
    // rewriter.setInsertionPointToEnd(currentBlock);

    auto countSmemObj = LLVM::getSharedMemoryObjectFromStruct(
      op.getLoc(), adaptor.getCount(),
      typeConverter->convertType(op.getCount().getType().getElementType()),
      rewriter);

    auto phaseSmemObj = LLVM::getSharedMemoryObjectFromStruct(
      op.getLoc(), adaptor.getPhase(),
      typeConverter->convertType(op.getPhase().getType().getElementType()),
      rewriter);

    auto wrapAroundVal = LLVM::createConstantI32(loc, rewriter, 1);

    auto dsDecRtnIntrincis = "llvm.amdgcn.ds_dec_rtn_u32";
    auto countBaseAddr = countSmemObj.getBase();
    auto countElemType = countSmemObj.getBaseElemType();
    SmallVector<Value, 6> args{countBaseAddr, wrapAroundVal};
    auto dsDecRtnOp = LLVM::createLLVMIntrinsicCallOp(rewriter, loc, dsDecRtnIntrincis, countElemType, args);
    auto res = dsDecRtnOp.getResult(0);
    llvm::errs() << *currentBlock;

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::AMD::populateSPMDOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
  patterns.add<ArriveBarrierOpConversion>(typeConverter, benefit);
}
