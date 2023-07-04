#ifndef FEATURE_UTILS_H
#define FEATURE_UTILS_H
#include <tvm/tir/transform.h>
#include <tvm/meta_schedule/feature_extractor.h>

#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../utils.h"

namespace tvm {
namespace tir {

using support::NDIntSet;

/*! \brief Type for multi-dimensional index */
using MultiIndex = std::vector<PrimExpr>;
/*! \brief Vector of int64_t */
using IntVec = std::vector<int64_t>;
/*! \brief Vector of for loops */
using ForVec = std::vector<const ForNode*>;

/*!
 * \brief An unordered_map for (for, buffer) => V
 * \tparam V The value type
 */
template <class V>
using ForBufferMap = std::unordered_map<const ForNode*, std::unordered_map<const BufferNode*, V>>;

/*! \brief Given x, compute log2(|x| + 1) */
inline double slog(double x) { return x >= 0 ? std::log2(x + 1) : std::log2(-x + 1); }

namespace utils{
    std::vector<int64_t> GetBufferShape(const Buffer& buffer, arith::Analyzer* analyzer);

    int64_t GetPragmaAutoUnroll(const ForNode* loop);

    int64_t FirstLoopExtent(const ForVec& loops, int64_t default_value);

    IntVec RelaxAndUnion(const std::vector<MultiIndex>& multi_indices, int64_t* numel,
                        arith::Analyzer* analyzer);

    int64_t GetVarStride(const std::vector<MultiIndex>& multi_indices, const IntVec& buffer_stride,
                        const Var& var);

    runtime::NDArray AsNDArray(const std::vector<std::vector<double>>& src);
} // namespace utils

namespace transform{
    Pass SimplifyForFeatureExtraction();
    Sequential PassListForPerStoreFeature();
} // namespace transform


/*! \brief A data structure managing loop nests */
struct LoopNest {
  int64_t prod = 1;    // The product of the extents of all the loops
  ForVec loops;        // All the loops
  IntVec auto_unroll;  // The loops with auto unroll pragma
  ForVec parallel;     // The loops whose ForKind are kParallel
  ForVec vectorize;    // The loops whose ForKind are kVectorized
  ForVec unroll;       // The loops whose ForKind are kUnrolled
  ForVec serial;       // The loops whose ForKind are kSerial
  ForVec blockIdx_x;   // The loops whose ForKind are kThreadBinding to blockIdx.x
  ForVec blockIdx_y;   // The loops whose ForKind are kThreadBinding to blockIdx.y
  ForVec blockIdx_z;   // The loops whose ForKind are kThreadBinding to blockIdx.z
  ForVec threadIdx_x;  // The loops whose ForKind are kThreadBinding to threadIdx.x
  ForVec threadIdx_y;  // The loops whose ForKind are kThreadBinding to threadIdx.y
  ForVec threadIdx_z;  // The loops whose ForKind are kThreadBinding to threadIdx.z
  ForVec vthread;      // The loops whose ForKind are kThreadBinding to vthread.*


  ForVec GetUnBindingLoop() {
    ForVec result;
    result.insert(result.end(), this->parallel.begin(), this->parallel.end());
    result.insert(result.end(), this->vectorize.begin(), this->vectorize.end());
    result.insert(result.end(), this->unroll.begin(), this->unroll.end());
    result.insert(result.end(), this->vthread.begin(), this->vthread.end());
    result.insert(result.end(), this->serial.begin(), this->serial.end());
    return result;
  }

  int64_t GetLoopsExtentProd(ForVec loops) {
    int64_t result = 1;
    for(auto l: loops) {
      VLOG(0) << l->extent;
      result = result * (*GetLoopIntExtent(l));
    }
    return result;
  }

  int64_t GetUnBindLoopsExtentProd() {
    return GetLoopsExtentProd(GetUnBindingLoop());
  }

  /*!
   * \brief Push a new loop into the loop nest
   * \param loop The loop to be pushed
   * \param auto_unroll_attr The auto unroll attribute of the loop
   * \return A list of for loops that the loop is bound to
   */
  ForVec* Push(const ForNode* loop, int64_t* auto_unroll_attr) {
    if (const int64_t* extent = GetLoopIntExtent(loop)) {
      this->prod *= *extent;
    }
    this->loops.push_back(loop);
    VLOG(0) << static_cast<int>(loop->kind);
    if ((*auto_unroll_attr = utils::GetPragmaAutoUnroll(loop)) > 0) {
      this->auto_unroll.push_back(*auto_unroll_attr);
    }
    ForVec* ref_loops = nullptr;
    if (loop->kind == ForKind::kParallel) {
      ref_loops = &parallel;
    } else if (loop->kind == ForKind::kVectorized) {
      ref_loops = &vectorize;
    } else if (loop->kind == ForKind::kUnrolled) {
      ref_loops = &unroll;
    } else if(loop->kind == ForKind::kSerial) {
      ref_loops = &serial;
    }else if (loop->kind == ForKind::kThreadBinding) {
      std::string thread_tag = loop->thread_binding.value()->thread_tag;
      if (thread_tag == "blockIdx.x") {
        ref_loops = &blockIdx_x;
      } else if (thread_tag == "blockIdx.y") {
        ref_loops = &blockIdx_y;
      } else if (thread_tag == "blockIdx.z") {
        ref_loops = &blockIdx_z;
      } else if (thread_tag == "threadIdx.x") {
        ref_loops = &threadIdx_x;
      } else if (thread_tag == "threadIdx.y") {
        ref_loops = &threadIdx_y;
      } else if (thread_tag == "threadIdx.z") {
        ref_loops = &threadIdx_z;
      } else if (support::StartsWith(thread_tag, "vthread")) {
        ref_loops = &vthread;
      } else {
        LOG(FATAL) << "ValueError: Unable to recognize thread tag: " << thread_tag;
      }
    }
    if (ref_loops != nullptr) {
      ref_loops->push_back(loop);
    }
    return ref_loops;
  }

  /*!
   * \brief Pop the last loop from the loop nest
   * \param loop The loop to be popped
   * \param ref_loops The list of for loops that the loop is bound to
   * \param auto_unroll_attr The auto unroll attribute of the loop
   */
  void Pop(const ForNode* loop, ForVec* ref_loops, int auto_unroll_attr) {
    if (ref_loops) {
      ref_loops->pop_back();
    }
    if (auto_unroll_attr > 0) {
      this->auto_unroll.pop_back();
    }
    if (const int64_t* extent = GetLoopIntExtent(loop)) {
      this->prod /= *extent;
    }
    this->loops.pop_back();
  }
};


}
}




#endif
