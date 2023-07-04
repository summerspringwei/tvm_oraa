
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
            // TODO(Chunwei) Add number of elements computed by a block, and compute the ratio
            
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


namespace group1 {
    /*! \brief extract the feature of load and store for each block*/
    struct Feature {
        struct LaunchInfo
        {
            // load global to shared features (Assembly code: LDG2S)
            int64_t num_warps = 0;                      // The number of warps within a block
            int64_t num_blocks = 0;                     // The number of blocks
            int64_t num_allocated_shared_memory = 0;    // The allocated shared memory within a block, in KB
            int64_t num_tensor_register = 0;            // The number of Tensor core registers allocated, in Bytes
            int64_t num_register_per_thread = 0;        // The allocated register
            
            // Hardware related info
            int64_t num_streaming_processors = 1;       // The number of streaming processors for the GPU
            int64_t num_of_partitions_per_sm = 4;       // The number of SM partitions for each SM
            int64_t L2_cache_size = 1;                  // The shared L2 cache size of the GPU
            int64_t maximum_shared_memory_size = 1;     // The maximum shared memory / L1 for each SM
            const int64_t kSuitableWarpsPerBlock = 8;   // A heuristic number, we think 8 warps is a maximum value
            const int64_t kMaximumRegisterPerBlock = 65536;

            static constexpr int64_t kCount = 4;
            LaunchInfo() = default;
            LaunchInfo(const PrimFuncNode* func) {}

            void Export(std::vector<double>* v) const {
                double vs [] = {
                    num_warps / (float)kSuitableWarpsPerBlock, 
                    num_blocks / (float)(num_streaming_processors * num_of_partitions_per_sm),
                    num_allocated_shared_memory / (float)(maximum_shared_memory_size), 
                    num_tensor_register * num_warps / (float)kMaximumRegisterPerBlock
                };
                v->insert(v->end(), std::begin(vs), std::end(vs));
            }
        };
    };
};

} // namespace transform


/*! \brief The feature extracted */
struct Feature {
  const BufferNode* buffer = nullptr;
  int buffer_order = -1;
  std::unique_ptr<transform::group0::Feature> group1 = nullptr;
  bool operator<(const Feature& other) const { return buffer_order < other.buffer_order; }
};

class LoadVarExtractor : private ExprVisitor {
    public:
    static Buffer Extract(const PrimExpr& expr){
        LoadVarExtractor extractor;
        extractor(expr);
        return extractor.Buffer_;
    }

    void VisitExpr_(const BufferLoadNode* op) {
        this->Buffer_ = op->buffer;
        this->indices_ = op->indices;
    }

    private:
    Buffer Buffer_;
    Array<PrimExpr> indices_;
};

class PerBlockFeatureCollector : private StmtVisitor {
    public:
    static std::vector<Feature> Collect(const IRModule& mod) {
    PerBlockFeatureCollector collector(true);
    for (const auto& kv : mod->functions) {
      if (const PrimFuncNode* func = kv.second.as<PrimFuncNode>()) {
        collector.clear();
        collector.set_func_buffer_map(func->buffer_map);
        for(auto it: func->buffer_map) {
            VLOG(0) << it.first << " " << it.second;
        }
        collector(func->body);
      }
    }
    std::vector<Feature> result;
    // result.reserve(collector.buffer_features_.size());
    // for (auto& it : collector.buffer_features_) {
    //   Feature& feature = it.second;
    //     result.push_back(std::move(feature));
    //   }
    // }
    // std::sort(result.begin(), result.end());
    return result;
  }
    void clear() {
        this->func_buffer_map_.clear();
        this->shared_memory_map_.clear();
        this->tensor_register_map_.clear();
        this->global_memory_map_.clear();
    }

    void set_func_buffer_map(Map<tir::Var, Buffer> func_buffer_map){
        for(auto& it : func_buffer_map){
            this->func_buffer_map_.Set(it.second->data, it.second);
        }
    }

    PerBlockFeatureCollector(bool extract_hardware):extract_hardware_(extract_hardware) {}

private:
    void VisitStmt_(const ForNode* loop) final {
        int64_t auto_unroll;
        ForVec* for_vec = loop_nest_.Push(loop, &auto_unroll);
        StmtVisitor::VisitStmt_(loop);
        loop_nest_.Pop(loop, for_vec, auto_unroll);
    }

    void VisitStmt_(const BufferStoreNode* store) final {
        if (store->value->IsInstance<IntImmNode>() || store->value->IsInstance<FloatImmNode>()) {
            return;
        }
        auto dst_buff = store->buffer;
        auto src_buff = LoadVarExtractor::Extract(store->value);
        // 1. Count global memory -> shared memory
        {
            auto it_dst = shared_memory_map_.find(dst_buff->data);
            auto it_src = func_buffer_map_.find(src_buff->data);
            VLOG(0) << "Find dst_buff: " << dst_buff->data << "src_buff: " << src_buff;
            if (it_dst!=shared_memory_map_.end() && it_src != func_buffer_map_.end()) {
                // VLOG(0) << "Store to shared memory " << (*it_dst).first;
                // VLOG(0) << "Load from global memory " << (*it_src).first;
                auto prod = loop_nest_.GetUnBindLoopsExtentProd();
                VLOG(0) << " global memory: " << (*it_src).first << 
                    " -> shared memory: " << (*it_dst).first << " prod " << prod;
            }
        }
        // 2. Count shared memory -> tensor core register
        {

        }
        // 3. Count tensor core register -> shared memory

        // 4. Count shared memory -> global memory
        {
            auto it_dst = func_buffer_map_.find(src_buff->data);
            auto it_src = shared_memory_map_.find(dst_buff->data);
            if ((it_dst != func_buffer_map_.end()) && (it_src != shared_memory_map_.end())) {
                auto prod = loop_nest_.GetUnBindLoopsExtentProd();
                VLOG(0) << "shared memory: " << (*it_src).first << " -> global memory: " <<
                   (*it_dst).first << " prod " << prod;
            }
        }
        
        this->VisitExpr(store->value);
    }

    // void VisitStmt_(const DeclBufferNode* op) {
    //     this->VisitStmt(op->body);
    // }

    void VisitStmt_(const BlockNode* block) final {
        StmtVisitor::VisitStmt_(block);
        for (const Buffer& buffer : block->alloc_buffers) {
            HandleBufferAlloc(buffer);
        }
    }

    void VisitStmt_(const AllocateNode* op) {
        const auto* ptr_type = op->buffer_var->type_annotation.as<PointerTypeNode>();
        VLOG(0) << op->buffer_var << " " << ptr_type->storage_scope << " " << ptr_type->element_type;
        // Get Allocate memory type
        if("shared" == ptr_type->storage_scope){
            shared_memory_map_.insert(std::make_pair(op->buffer_var, GetRef<Allocate>(op)));
        }else if("wmma.accumulator" == ptr_type->storage_scope ||
            "wmma.matrix_b" == ptr_type->storage_scope ||
            "wmma.matrix_a" == ptr_type->storage_scope){
            tensor_register_map_.insert(std::make_pair(op->buffer_var, GetRef<Allocate>(op)));
        }else{
            LOG_WARNING << "Unrecognized storage_scope: " << ptr_type->storage_scope;
        }
        this->VisitStmt(op->body);
    }

    void HandleBufferAlloc(const Buffer& buffer){
        VLOG(0) << PrettyPrint(buffer).c_str();
    }

    std::unordered_map<Var, Allocate, ObjectPtrHash, ObjectPtrEqual> local_memory_map_;
    std::unordered_map<Var, Allocate, ObjectPtrHash, ObjectPtrEqual> tensor_register_map_;
    std::unordered_map<Var, Allocate, ObjectPtrHash, ObjectPtrEqual> shared_memory_map_;
    std::unordered_map<Var, Allocate, ObjectPtrHash, ObjectPtrEqual> global_memory_map_;
    Map<tir::Var, Buffer> func_buffer_map_;
    
    // std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> local_memory_map_;
    // std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> tensor_register_map_;
    // std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> shared_memory_map_;
    // std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> global_memory_map_;
    arith::Analyzer analyzer_;
    LoopNest loop_nest_ = {};
    bool extract_hardware_;
};

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
        tir::PerBlockFeatureCollector::Collect(mod);
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
