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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

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

void lowerInit(OpBuilder thenBuilder, Location loc, Value barrierCountView,
               Value barrierPhaseView, int initCount, Value phaseOp,
               Value threadId) {
  auto initCountOp =
      thenBuilder.create<arith::ConstantIntOp>(loc, initCount, 32);

  auto countTensorType = RankedTensorType::get({1}, initCountOp.getType());
  auto countTensorVal =
      thenBuilder.create<triton::SplatOp>(loc, countTensorType, initCountOp);

  auto phaseTensorType = RankedTensorType::get({1}, phaseOp.getType());
  auto phaseTensorVal =
      thenBuilder.create<triton::SplatOp>(loc, phaseTensorType, phaseOp);

  auto countStoreOp = thenBuilder.create<ttg::LocalStoreOp>(loc, countTensorVal,
                                                            barrierCountView);
  auto phaseStoreOp = thenBuilder.create<ttg::LocalStoreOp>(loc, phaseTensorVal,
                                                            barrierPhaseView);
}
Value getMBarrierPhaseBit(OpBuilder &builder, Operation *op,
                          bool emptyBarrier) {
  auto loc = op->getLoc();
  assert(isa<ttng::ProducerAcquireOp>(op) || isa<ttng::ConsumerWaitOp>(op));
  Value curPhase;
  if (auto acq = dyn_cast<ttng::ProducerAcquireOp>(op))
    curPhase = acq.getPhase();
  else if (auto wait = dyn_cast<ttng::ConsumerWaitOp>(op))
    curPhase = wait.getPhase();
  if (emptyBarrier) {
    // curPhase = curPhase xor True for emptyBarrier.
    Value _1_1b = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    curPhase = builder.create<mlir::arith::XOrIOp>(loc, curPhase, _1_1b);
  }
  // LLVM_DEBUG(curPhase.dump());
  return curPhase;
}

void processProducerAcquireOp(OpBuilder &builder, ttng::ProducerAcquireOp op,
                              Value bufferEmptyPhaseView) {
  auto loc = op.getLoc();
  Value localPhase = getMBarrierPhaseBit(builder, op, true);
  auto i32Ty = builder.getIntegerType(32);
  localPhase = builder.create<arith::ExtUIOp>(loc, i32Ty, localPhase);
  auto phaseTensorType = RankedTensorType::get({1}, builder.getI32Type());
  auto initialCondition = localPhase;
  // auto whileOp = builder.create<scf::WhileOp>(loc, initialCondition.getType(),
  //                                             initialCondition);
  auto whileOp = builder.create<scf::WhileOp>(loc, TypeRange{}, ValueRange{});
  // Before block
  // Before block
  Block *beforeBlock = builder.createBlock(&whileOp.getBefore());
  // auto conditionArg = beforeBlock->addArgument(i32Ty, loc);
  builder.setInsertionPointToEnd(beforeBlock);

  Value barrierPhaseTensor = builder.create<ttg::LocalLoadOp>(
      loc, phaseTensorType, bufferEmptyPhaseView);
  auto index_0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  // SmallVector<Value> index;
  // index.push_back(index_0);
  Value barrierPhase =
      builder.create<mlir::tensor::ExtractOp>(loc, barrierPhaseTensor, ValueRange{index_0});
  Value phaseCond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  barrierPhase, localPhase);
  // builder.create<scf::ConditionOp>(loc, phaseCond, beforeBlock->getArguments());
  builder.create<scf::ConditionOp>(loc, phaseCond, ValueRange{});

  // after block
  Block *afterBlock = builder.createBlock(&whileOp.getAfter());
  builder.setInsertionPointToEnd(afterBlock);
  auto instrinsic = "llvm.amdgcn.s_sleep 10";
  auto sleepOp = LLVM::createLLVMIntrinsicCallOp(builder, loc, instrinsic, TypeRange{}, ValueRange{});
  builder.create<scf::YieldOp>(loc,  ValueRange{});
}

static const int THREADS_PER_TASK = 64;
static const int WAVES_PER_TASK = 4;
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

    Type barrierElementMemDescType =
        tt::MemDescType::get({1}, builder.getI32Type(), barrierEncoding,
                             sharedMemorySpace, /*mutableMemory=*/true);

    Value bufferFullArray = builder.create<mlir::triton::gpu::LocalAllocOp>(
        loc, barrierMemDescType, Value());
    Value bufferEmptyArray = builder.create<mlir::triton::gpu::LocalAllocOp>(
        loc, barrierMemDescType, Value());

    auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    auto one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
    // Helper function for extracting one element
    auto createFieldView = [&](OpBuilder builder, Location loc, Value array,
                               Value barrierIndexOp,
                               Value fieldIndex) -> Value {
      SmallVector<Value> elementIdx({barrierIndexOp, fieldIndex});
      return builder.create<ttg::MemDescSubviewOp>(
          loc, barrierElementMemDescType, array, elementIdx);
    };

    // Initialize the barriers
    //  TBD: Check init counts
    // If thread0, set barrier to initCount
    auto i32ty = builder.getIntegerType(32);
    auto threadId = builder.create<ROCDL::ThreadIdXOp>(loc, i32ty);
    Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               threadId, zero);
    auto ifOp = builder.create<scf::IfOp>(loc, cond);
    auto thenBuilder = ifOp.getThenBodyBuilder();
    Value phaseInitValue = zero;
    Value phaseOffset = one;
    Value countOffset = zero;

    for (unsigned barrierIndex = 0; barrierIndex < createTokenOp.getNum();
         barrierIndex++) {
      Value barrierIndexOp =
          thenBuilder.create<arith::ConstantIntOp>(loc, barrierIndex, 32);
      Value barrierFullCountView = createFieldView(
          thenBuilder, loc, bufferFullArray, barrierIndexOp, countOffset);
      Value barrierFullPhaseView = createFieldView(
          thenBuilder, loc, bufferFullArray, barrierIndexOp, phaseOffset);
      lowerInit(thenBuilder, loc, barrierFullCountView, barrierFullPhaseView,
                WAVES_PER_TASK, phaseInitValue, threadId);

      Value barrierEmptyCountView = createFieldView(
          thenBuilder, loc, bufferEmptyArray, barrierIndexOp, countOffset);
      Value barrierEmptyPhaseView = createFieldView(
          thenBuilder, loc, bufferEmptyArray, barrierIndexOp, phaseOffset);
      lowerInit(thenBuilder, loc, barrierEmptyCountView, barrierEmptyPhaseView,
                WAVES_PER_TASK, phaseInitValue, threadId);
      eraseOps.insert(createTokenOp);
    }
    builder.create<mlir::gpu::BarrierOp>(loc);

    for (Operation *user : createTokenOp.getResult().getUsers()) {
      auto loc = user->getLoc();
      builder.setInsertionPoint(user);
      if (auto op = dyn_cast<ttng::ProducerAcquireOp>(user)) {
        Value bufferEmptyPhase = createFieldView(builder, loc, bufferEmptyArray,
                                                 op.getIdx(), phaseOffset);
        processProducerAcquireOp(builder, op, bufferEmptyPhase);
        eraseOps.insert(op);
      }
    }
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
    // llvm::errs() << "pre: " << mod << "\n";
    lowerGetAsyncTaskIdOp(mod);
    lowerTokenOperations(mod);
    llvm::errs() << "post: " << mod << "\n";
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createTritonAMDGPUWSLoweringPass() {
  return std::make_unique<TritonAMDGPUWSLoweringPass>();
}
