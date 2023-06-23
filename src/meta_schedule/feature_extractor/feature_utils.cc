
#include "feature_utils.h"

namespace tvm {
namespace tir {
namespace utils {

/*!
 * \brief Get the shape of the buffer
 * \param buffer The buffer
 * \param analyzer The analyzer
 * \return The shape of the buffer
 */
std::vector<int64_t> GetBufferShape(const Buffer& buffer, arith::Analyzer* analyzer) {
  int ndim = buffer->shape.size();
  std::vector<int64_t> result;
  result.reserve(ndim);
  for (const PrimExpr& i : buffer->shape) {
    if (const IntImmNode* int_imm = i.as<IntImmNode>()) {
      result.push_back(int_imm->value);
      continue;
    }
    arith::ConstIntBound bound = analyzer->const_int_bound(i);
    if (0 <= bound->max_value && bound->max_value < arith::ConstIntBound::kPosInf) {
      result.push_back(bound->max_value);
    } else {
      result.push_back(1);
    }
  }
  return result;
}

/*!
 * \brief Given a loop, return its `pragma_auto_unroll_max_step` annotation if it exists
 * \param loop The loop to be checked
 * \return The value of `pragma_auto_unroll_max_step` if it exists, or -1 if it does not exist
 */
int64_t GetPragmaAutoUnroll(const ForNode* loop) {
  if (Optional<IntImm> auto_unroll = GetAnn<IntImm>(loop, tir::attr::pragma_auto_unroll_max_step)) {
    return auto_unroll.value()->value;
  }
  return -1;
}

/*!
 * \brief Given a list of loops, return the extent of the first loop if the list is not empty,
 * and the first loop has constant extent. Otherwise returns the default value given
 * \param loops The list of loops to be checked
 * \param default_value The default value to be returned if the list is empty or the first loop
 * does not have constant extent
 * \return The extent of the first loop if the list is not empty, or the first loop has constant
 * extent. Otherwise returns the default value
 */
int64_t FirstLoopExtent(const ForVec& loops, int64_t default_value) {
  if (!loops.empty()) {
    if (const int64_t* extent = GetLoopIntExtent(loops[0])) {
      return *extent;
    }
  }
  return default_value;
}

/*!
 * \brief Relax each of the multi-indexing pattern according to the domains bound in the analyzer,
 * and then union them into a single region
 * \param multi_index_pattern A list of multi-index pattern to be relaxed
 * \param numel The size of the single region after union
 * \param analyzer The analyzer that contains the domain information
 * \return The relaxed and unioned region
 */
IntVec RelaxAndUnion(const std::vector<MultiIndex>& multi_indices, int64_t* numel,
                     arith::Analyzer* analyzer) {
  *numel = 1;
  if (multi_indices.empty()) {
    return {};
  }
  int n_indices = multi_indices.size();
  int ndim = multi_indices[0].size();
  IntVec access_shape(ndim, 0);
  for (int i = 0; i < ndim; ++i) {
    int64_t minimum = arith::ConstIntBound::kPosInf;
    int64_t maximum = arith::ConstIntBound::kNegInf;
    for (int j = 0; j < n_indices; ++j) {
      arith::ConstIntBound bound = analyzer->const_int_bound(multi_indices[j][i]);
      minimum = std::min(minimum, bound->min_value);
      maximum = std::max(maximum, bound->max_value);
    }
    *numel *= maximum - minimum + 1;
    access_shape[i] = maximum - minimum + 1;
  }
  return access_shape;
}

/*!
 * \brief Given a list of multi-index pattern, return the minimal stride of a variable on it
 * \param multi_indices The list of multi-index pattern
 * \param buffer_stride The stride of the buffer
 * \param var The variable to be checked
 * \return The minimal stride of the variable on the multi-index pattern
 */
int64_t GetVarStride(const std::vector<MultiIndex>& multi_indices, const IntVec& buffer_stride,
                     const Var& var) {
  class CoefficientExtractor : private ExprVisitor {
   public:
    static int64_t Extract(const PrimExpr& expr, const Var& var) {
      CoefficientExtractor extractor(var);
      extractor.VisitExpr(expr);
      return (extractor.visited_var && !extractor.visited_mul && !extractor.visited_add)
                 ? 1
                 : (extractor.visited_var ? extractor.stride : 0);
    }

   private:
    explicit CoefficientExtractor(const Var& var)
        : var(var), stride(0), visited_var(false), visited_add(false), visited_mul(false) {}

    void VisitExpr_(const MulNode* node) override {
      ExprVisitor::VisitExpr_(node);
      if (visited_var && !visited_add) {
        if (const auto* a = node->a.as<IntImmNode>()) {
          visited_mul = true;
          stride = a->value;
        } else if (const auto* b = node->b.as<IntImmNode>()) {
          visited_mul = true;
          stride = b->value;
        }
      }
    }

    void VisitExpr_(const AddNode* node) override {
      ExprVisitor::VisitExpr_(node);
      if (visited_var && !visited_mul) {
        visited_add = true;
        stride = 1;
      }
    }

    void VisitExpr_(const VarNode* node) override {
      if (node == var.get()) {
        visited_var = true;
        stride = 2;
      }
    }

    const Var& var;
    int64_t stride;
    bool visited_var;
    bool visited_add;
    bool visited_mul;
  };

  constexpr int64_t kNotFound = std::numeric_limits<int64_t>::max();
  int ndim = buffer_stride.size();
  // Calculate the min stride possible
  int64_t result = kNotFound;
  for (const MultiIndex& multi_index : multi_indices) {
    ICHECK_EQ(multi_index.size(), buffer_stride.size());
    // Find the rightest dimension that contains the given variable
    for (int i = ndim - 1; i >= 0; --i) {
      int64_t coef = CoefficientExtractor::Extract(multi_index[i], var);
      if (coef != 0) {
        result = std::min(result, std::abs(coef) * buffer_stride[i]);
        break;
      }
    }
  }
  return (result == kNotFound) ? 0 : result;
}

/*!
 * \brief Converts a 2-dimensional STL vector to a TVM NDArray
 * \param src The source 2-dimensional STL vector
 * \return The converted TVM NDArray
 */
runtime::NDArray AsNDArray(const std::vector<std::vector<double>>& src) {
  ICHECK(!src.empty());
  int n = src.size();
  int m = src[0].size();
  runtime::NDArray tgt = runtime::NDArray::Empty(
      /*shape=*/{n, m},
      /*dtype=*/DLDataType{kDLFloat, 64, 1},
      /*ctx=*/DLDevice{kDLCPU, 0});
  double* data = static_cast<double*>(tgt->data);
  for (const std::vector<double>& row : src) {
    for (double v : row) {
      *data++ = v;
    }
  }
  return tgt;
}

}  // namespace utils

namespace transform {

/*!
 * \brief Create a pass that simplifies the IR for feature extraction
 * \return The pass created
 */
Pass SimplifyForFeatureExtraction() {
  class Simplifier : private StmtExprMutator {
   public:
    static Stmt Run(Stmt stmt) { return Simplifier()(std::move(stmt)); }

   private:
    static bool HasBufferLoad(const PrimExpr& expr) {
      bool found = false;
      PostOrderVisit(expr, [&found](const ObjectRef& node) {
        if (node->IsInstance<BufferLoadNode>()) {
          found = true;
        }
      });
      return found;
    }

    PrimExpr VisitExpr_(const SelectNode* node) final {
      if (HasBufferLoad(node->true_value) || HasBufferLoad(node->false_value) ||
          HasBufferLoad(node->condition)) {
        return GetRef<Select>(node);
      }
      return make_const(node->dtype, 1.0);
    }

    PrimExpr VisitExpr_(const VarNode* var) final {
      if (unit_vars_.count(GetRef<Var>(var))) {
        return make_const(var->dtype, 0.0);
      }
      return GetRef<Var>(var);
    }

    Stmt VisitStmt_(const ForNode* loop) final {
      if (is_zero(loop->min) && is_one(loop->extent) && loop->kind == ForKind::kSerial &&
          loop->annotations.empty()) {
        unit_vars_.insert(loop->loop_var);
        return VisitStmt(loop->body);
      } else {
        return StmtExprMutator::VisitStmt_(loop);
      }
    }

    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> unit_vars_;
  };
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    PrimFuncNode* n = f.CopyOnWrite();
    n->body = Simplifier::Run(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.SimplifyForFeatureExtraction", {});
}

/*!
 * \brief Create a list of passes that preprocesses the IR for feature extraction
 * \return The list of passes created
 */
Sequential PassListForPerStoreFeature() {
  return Sequential({
      tir::transform::RemoveWeightLayoutRewriteBlock(/*skip_ndarray_rewrite*/ true),
      tir::transform::SimplifyForFeatureExtraction(),
      tir::transform::LowerCrossThreadReduction(),
      tir::transform::LowerInitBlock(),
      tir::transform::PlanAndUpdateBufferAllocationLocation(),
      tir::transform::ConvertBlocksToOpaque(),
      tir::transform::UnifyThreadBinding(),
      tir::transform::CompactBufferAllocation(),
      tir::transform::LowerMatchBuffer(),
      tir::transform::Simplify(),
  });
}

} // namespace transform
} // namespace tir
} // namespace tvm
