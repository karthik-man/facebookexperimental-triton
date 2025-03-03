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

    auto countSmemObj = LLVM::getSharedMemoryObjectFromStruct(
      op.getLoc(), adaptor.getCount(),
      typeConverter->convertType(op.getCount().getType().getElementType()),
      rewriter);

    auto phaseSmemObj = LLVM::getSharedMemoryObjectFromStruct(
      op.getLoc(), adaptor.getPhase(),
      typeConverter->convertType(op.getPhase().getType().getElementType()),
      rewriter);

    auto wrapAroundVal = LLVM::createConstantI32(loc, rewriter, 1);


    GCNBuilder gcnBuilder;
    auto &dec_rtn = *gcnBuilder.create("ds_dec_rtn_u32");
    auto retVal = gcnBuilder.newOperand("=v");
    auto countBaseAddr = gcnBuilder.newOperand(countSmemObj.getBase(), "v");
    auto wav = gcnBuilder.newOperand(wrapAroundVal, "v");
    dec_rtn(retVal, countBaseAddr, wav);
    auto &wait_cnt = *gcnBuilder.create("s_waitcnt lgkmcnt(0)");
    wait_cnt();
    auto res = gcnBuilder.launch(rewriter, loc, i32_ty, true /*hasSideEffects*/);

    Value zero = i32_val(0);
    Value allArrived = icmp_eq(res, zero);

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterPhaseFlipBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *phaseFlipBlock = rewriter.createBlock(afterPhaseFlipBlock);
    rewriter.setInsertionPointToEnd(currentBlock);

    rewriter.create<LLVM::CondBrOp>(loc, allArrived , phaseFlipBlock,
                                    afterPhaseFlipBlock);

    rewriter.setInsertionPointToStart(phaseFlipBlock);
    auto phaseBaseAddr = phaseSmemObj.getBase();
    GCNBuilder gcnBuilder1;
    Value one = i32_val(1);
    auto &xor_phase = *gcnBuilder1.create("ds_xor_b32");
    auto baseAddrArg = gcnBuilder1.newOperand(phaseBaseAddr, "v");
    auto oneArg = gcnBuilder1.newOperand(one, "v");
    xor_phase(baseAddrArg, oneArg);
    gcnBuilder1.launch(rewriter, loc, i32_ty, true /*hasSideEffects*/);

    auto br = rewriter.create<LLVM::BrOp>(loc, afterPhaseFlipBlock);
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
