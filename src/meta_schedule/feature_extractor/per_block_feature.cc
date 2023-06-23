
#include "feature_utils.h"
#include "../utils.h"


namespace tvm {
namespace tir {

namespace transform {
/*!
 * \brief Create a list of passes that preprocesses the IR for feature extraction
 * \return The list of passes created
 */
Sequential PassListForPerBlockFeature() {
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
      tir::transform::InjectSoftwarePipeline(),
      tir::transform::LowerOpaqueBlock(),
  });
}

namespace group0 {
    /*! \brief extract the feature of load and store for each block*/
    struct Feature {
        struct LoadStoreOps
        {
            // load global to shared features (Assembly code: LDG2S)
            int64_t load_global_to_shared_bytes = 0;        // The number of bytes load from global memory to shared memory
            int64_t load_global_to_shared_transactions = 0;  // The number of warps(memory transactions) to be executed
            int64_t load_stride_for_global = 0;            // The stride for consucutive threads when load from global memory (sizeof(T) is best, otherwise real value)
            int64_t store_stride_for_shared = 0;            // The stride for consucutive thread when store to shared memory
            // Store from shared to global (Assembly Code: STG.EXXX)
            int64_t store_shared_to_global_bytes = 0;
            int64_t store_shared_to_global_transactions = 0;
            int64_t store_stride_for_global = 0;
            int64_t load_stride_for_shared = 0;
            // TODO(Chunwei) Add shared memory to/from register

            // TODO(Chunwei) Add register to/from global memory

            static constexpr int64_t kCount = 8;
            LoadStoreOps() = default;
            LoadStoreOps(const PrimFuncNode* func) {}

            void Export(std::vector<double>* v) const {
                double vs [] = {
                    load_global_to_shared_bytes, load_global_to_shared_transactions,
                    load_stride_for_global, store_stride_for_shared,
                    store_shared_to_global_bytes, store_shared_to_global_transactions,
                    store_stride_for_global, load_stride_for_shared
                };
                v->insert(v->end(), std::begin(vs), std::end(vs));
            }
        };
    };

    class LoadStoreCounter : private StmtVisitor {
        std::stack<ForNode*> loop_stack = {};
        private:
        void VisitStmt_(const ForNode* loop) final {
            int64_t auto_unroll;
            ForVec* for_vec = loop_nest_.Push(loop, &auto_unroll);
            StmtVisitor::VisitStmt_(loop);
            loop_nest_.Pop(loop, for_vec, auto_unroll);
        }

        void VisitStmt_(const BlockNode* block) final {
            StmtVisitor::VisitStmt_(block);
            for(auto buffer: block->alloc_buffers){

            }
        }
        arith::Analyzer analyzer_;
        LoopNest loop_nest_ = {};
    };
};


/*! \brief The feature extracted */
struct Feature {
  const BlockNode* block = nullptr;
  

  bool operator<(const Feature& other) const { return false; }
};
} // namespace transform
} // namespace tir
} // namespace tvm

namespace tvm {
namespace meta_schedule {

class PerBlockFeatureNode : public FeatureExtractorNode {
public:
    bool extract_workload;
    int feature_vector_length;
    void VisitAttrs(AttrVisitor* v) {
        v->Visit("feature_vector_length", &feature_vector_length);
    }

    void ExtractSingle(IRModule mod, std::vector<std::vector<double>>* results) {
        static transform::Sequential passes = tir::transform::PassListForPerBlockFeature();
        mod = passes(std::move(mod));
        VLOG(0) << PrettyPrint(mod);
    }
    
    Array<runtime::NDArray> ExtractFrom(const TuneContext& tune_context,
                                      const Array<MeasureCandidate>& candidates) {
        std::vector<runtime::NDArray> results;
        auto f = [this, &candidates, &results](int, int task_id) -> void {
        const auto& candidate = candidates[task_id];
        std::vector<std::vector<double>> features;
        ExtractSingle(DeepCopyIRModule(candidate->sch->mod()), &features);
        // results[task_id] = tir::utils::AsNDArray(features);
    };
    support::parallel_for_dynamic(0, candidates.size(), tune_context->num_threads, f);
        return results;
  }

  static constexpr const char* _type_key = "meta_schedule.PerBlockFeature";
  TVM_DECLARE_FINAL_OBJECT_INFO(PerBlockFeatureNode, FeatureExtractorNode);
};


FeatureExtractor FeatureExtractor::PerBlockFeature(bool extract_workload) {
  ObjectPtr<PerBlockFeatureNode> n = make_object<PerBlockFeatureNode>();
  n->extract_workload = extract_workload;
  n->feature_vector_length = 0; // TODO
  if (extract_workload) {
    n->feature_vector_length += 0; // TODO
  }
  return FeatureExtractor(n);
}

TVM_REGISTER_NODE_TYPE(PerBlockFeatureNode);
TVM_REGISTER_GLOBAL("meta_schedule.FeatureExtractorPerBlockFeature")
    .set_body_typed(FeatureExtractor::PerBlockFeature);

} // namespace meta_schedule
} // namespace tvm
