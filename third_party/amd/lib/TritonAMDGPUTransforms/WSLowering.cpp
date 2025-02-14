#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include <set>

#include "mlir/IR/OperationSupport.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;
namespace ttng = ::mlir::triton::nvidia_gpu;

namespace {

void lowerGetAsyncTaskIdOp(Operation *parentOp) {
  DenseSet<Operation *> eraseOps;
  parentOp->walk([&](ttng::GetAsyncTaskIdOp op) {
    auto loc = op.getLoc();
    OpBuilder builder(op);
    auto i32ty = builder.getIntegerType(32);
    auto workIDX = builder.create<ROCDL::ThreadIdXOp>(loc, i32ty);
    auto constWaveSize = builder.create<arith::ConstantIntOp>(loc, 64, 32);
    auto warpIDX = builder.create<arith::DivSIOp>(loc, workIDX, constWaveSize);
    op.getResult().replaceAllUsesWith(warpIDX);
    eraseOps.insert(op);
  });
  for (Operation *op : eraseOps)
    op->erase();
}

void lowerInit(OpBuilder builder, Location loc, Value barrierCountView, Value barrierPhaseView, int initCount, int phase) {
  //If thread0, set barrier to initCount
  auto i32ty = builder.getIntegerType(32);
  auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  auto initCountOp = builder.create<arith::ConstantIntOp>(loc, initCount, 32);
  auto phaseOp = builder.create<arith::ConstantIntOp>(loc, phase, 32);
  auto threadId = builder.create<ROCDL::ThreadIdXOp>(loc, i32ty);
  Value cond = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, threadId, zero);
  auto ifOp = builder.create<scf::IfOp>(loc, cond);
  auto thenBuilder = ifOp.getThenBodyBuilder();
  auto countStoreOp =
      thenBuilder.create<ttg::LocalStoreOp>(loc, initCountOp, barrierCountView);
  auto phaseStoreOp =
      thenBuilder.create<ttg::LocalStoreOp>(loc, phaseOp, barrierPhaseView);

}

static const int THREADS_PER_TASK = 64;
void lowerTokenOperations(Operation *parentOp) {
  DenseSet<Operation *> eraseOps;
  parentOp->walk([&](ttng::CreateTokenOp createTokenOp) {
    MLIRContext *context = createTokenOp.getContext();
    OpBuilder builder(createTokenOp);
    Location loc = createTokenOp.getLoc();

    Attribute sharedMemorySpace =
        triton::gpu::SharedMemorySpaceAttr::get(context);
    auto barrierCTALayout =
        ttg::CTALayoutAttr::get(context, /*CTAsPerCGA=*/{1},
                                /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
    auto barrierEncoding =
        ttg::SharedEncodingAttr::get(context, 1, 1, 1, {0}, barrierCTALayout);
    
    Type barrierMemDescType =
        tt::MemDescType::get({createTokenOp.getNum(), 2}, builder.getI32Type(),
                             barrierEncoding, sharedMemorySpace,
                             /*mutableMemory=*/true);

    Type barrierElementMemDescType = tt::MemDescType::get({1}, builder.getI32Type(), barrierEncoding,
                             sharedMemorySpace, /*mutableMemory=*/true);

    Value bufferFullArray = builder.create<mlir::triton::gpu::LocalAllocOp>(
        loc, barrierMemDescType, Value());
    Value bufferEmptyArray = builder.create<mlir::triton::gpu::LocalAllocOp>(
        loc, barrierMemDescType, Value());

    auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    auto one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
    //Helper function for extracting one element
    auto createFieldView = [&](Location loc, Value array, int barrierIndex, Value fieldIndex) -> Value {
      Value barrierIndexOp = builder.create<arith::ConstantIntOp>(loc, barrierIndex, 32);
      SmallVector<Value> elementIdx({barrierIndexOp, fieldIndex});
      return builder.create<ttg::MemDescSubviewOp>(loc, barrierElementMemDescType, array, elementIdx);
    };
    //Initialize the barriers
    for (unsigned barrierIndex = 0; barrierIndex < createTokenOp.getNum(); barrierIndex++) {
      Value barrierFullCountView = createFieldView(loc, bufferFullArray, barrierIndex, zero);
      Value barrierFullPhaseView = createFieldView(loc, bufferFullArray, barrierIndex, one);
      lowerInit(builder, loc, barrierFullCountView, barrierFullPhaseView, THREADS_PER_TASK, 0);

      Value barrierEmptyCountView = createFieldView(loc, bufferEmptyArray, barrierIndex, zero);
      Value barrierEmptyPhaseView = createFieldView(loc, bufferEmptyArray, barrierIndex, one);
      lowerInit(builder, loc, barrierEmptyCountView, barrierEmptyPhaseView, THREADS_PER_TASK, 0);
      eraseOps.insert(createTokenOp);
    }
    builder.create<mlir::gpu::BarrierOp>(loc);

  });
  
  for (Operation *op : eraseOps)
    op->erase();
}



class TritonAMDGPUWSLoweringPass
    : public TritonAMDGPUWSLoweringBase<TritonAMDGPUWSLoweringPass> {
public:
  TritonAMDGPUWSLoweringPass() = default;
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    llvm::errs() << "pre: " << mod << "\n";
    lowerGetAsyncTaskIdOp(mod);
    lowerTokenOperations(mod);
    llvm::errs() << "post: " << mod << "\n";
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createTritonAMDGPUWSLoweringPass() {
  return std::make_unique<TritonAMDGPUWSLoweringPass>();
}
