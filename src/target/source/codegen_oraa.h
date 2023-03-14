
/*!
 * \brief A base class to generate ORAA code.
 *
 *  CodeGenORAA generates Python like code.
 *
 */

#ifndef TVM_TARGET_CODEGEN_ORAA_H
#define TVM_TARGET_CODEGEN_ORAA_H

#include <tvm/ir/op.h>
#include <tvm/target/codegen.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include "../../tir/transforms/ir_utils.h"
#include "codegen_source_base.h"

namespace tvm {
namespace codegen {

using namespace tir;

/**
 * \brief A class to generate ORAA code.
*/
class CodeGenORAA : public ExprFunctor<void(const PrimExpr&, std::ostream&)>,
                 public StmtFunctor<void(const Stmt&)>,
                 public CodeGenSourceBase {

public:
  /*!
   * \brief Initialize the code generator.
   * \param output_ssa Whether output SSA.
   */
  void Init();
  /*!
   * \brief Add the function to the generated module.
   * \param f The function to be compiled.
   * \param whether to append return 0 in the end.
   */
  void AddFunction(const PrimFunc& f);
  /*!
   * \brief Finalize the compilation and return the code.
   * \return The code.
   */
  virtual std::string Finish();
  /*!
   * \brief Print the Stmt n to CodeGenC->stream
   * \param n The statement to be printed.
   */
  void PrintStmt(const Stmt& n) { VisitStmt(n); }
  /*!
   * \brief Print the expression n(or its ssa id if in ssa mode) into os
   * \param n The expression to be printed.
   * \param os The output stream
   */
  void PrintExpr(const PrimExpr& n, std::ostream& os);
  /*!
   * \brief Same as PrintExpr, but simply returns result string
   * \param n The expression to be printed.
   */
  std::string PrintExpr(const PrimExpr& n) {
    std::ostringstream os;
    PrintExpr(n, os);
    return os.str();
  }
  // The following parts are overloadable print operations.
  /*!
   * \brief Print the function header before the argument list
   * \param os The output stream
   *
   *  Example: stream << "def";
   */
  virtual void PrintFuncPrefix(std::ostream& os);  // NOLINT(*)
  /*!
   * \brief Print the final return at the end the function.
   */
  virtual void PrintFinalReturn();  // NOLINT(*)
  /*!
   * \brief Initialize codegen state for generating f.
   * \param f The function to be compiled.
   */
  virtual void InitFuncState(const PrimFunc& f);
  // expression
  void VisitExpr_(const VarNode* op, std::ostream& os);         // NOLINT(*)
  void VisitExpr_(const LoadNode* op, std::ostream& os);        // NOLINT(*)
  void VisitExpr_(const BufferLoadNode* op, std::ostream& os);  // NOLINT(*)
  void VisitExpr_(const LetNode* op, std::ostream& os);         // NOLINT(*)
  void VisitExpr_(const CallNode* op, std::ostream& os);        // NOLINT(*)
  void VisitExpr_(const AddNode* op, std::ostream& os);         // NOLINT(*)
  void VisitExpr_(const SubNode* op, std::ostream& os);         // NOLINT(*)
  void VisitExpr_(const MulNode* op, std::ostream& os);         // NOLINT(*)
  void VisitExpr_(const DivNode* op, std::ostream& os);         // NOLINT(*)
  void VisitExpr_(const ModNode* op, std::ostream& os);         // NOLINT(*)
  void VisitExpr_(const MinNode* op, std::ostream& os);         // NOLINT(*)
  void VisitExpr_(const MaxNode* op, std::ostream& os);         // NOLINT(*)
  void VisitExpr_(const EQNode* op, std::ostream& os);          // NOLINT(*)
  void VisitExpr_(const NENode* op, std::ostream& os);          // NOLINT(*)
  void VisitExpr_(const LTNode* op, std::ostream& os);          // NOLINT(*)
  void VisitExpr_(const LENode* op, std::ostream& os);          // NOLINT(*)
  void VisitExpr_(const GTNode* op, std::ostream& os);          // NOLINT(*)
  void VisitExpr_(const GENode* op, std::ostream& os);          // NOLINT(*)
  void VisitExpr_(const AndNode* op, std::ostream& os);         // NOLINT(*)
  void VisitExpr_(const OrNode* op, std::ostream& os);          // NOLINT(*)
  void VisitExpr_(const CastNode* op, std::ostream& os);        // NOLINT(*)
  void VisitExpr_(const NotNode* op, std::ostream& os);         // NOLINT(*)
  void VisitExpr_(const SelectNode* op, std::ostream& os);      // NOLINT(*)
  void VisitExpr_(const RampNode* op, std::ostream& os);        // NOLINT(*)
  void VisitExpr_(const ShuffleNode* op, std::ostream& os);     // NOLINT(*)
  void VisitExpr_(const BroadcastNode* op, std::ostream& os);   // NOLINT(*)
  void VisitExpr_(const IntImmNode* op, std::ostream& os);      // NOLINT(*)
  void VisitExpr_(const FloatImmNode* op, std::ostream& os);    // NOLINT(*)
  void VisitExpr_(const StringImmNode* op, std::ostream& os);   // NOLINT(*)
  // statment
  void VisitStmt_(const LetStmtNode* op);
  void VisitStmt_(const StoreNode* op);
  void VisitStmt_(const BufferStoreNode* op);
  void VisitStmt_(const ForNode* op);
  void VisitStmt_(const WhileNode* op);
  void VisitStmt_(const IfThenElseNode* op);
  void VisitStmt_(const AllocateNode* op);
  void VisitStmt_(const AttrStmtNode* op);
  void VisitStmt_(const AssertStmtNode* op);
  void VisitStmt_(const EvaluateNode* op);
  void VisitStmt_(const SeqStmtNode* op);
  void VisitStmt_(const AllocateConstNode* op);
  void VisitStmt_(const DeclBufferNode* op);
  /*!
   * \brief Print expr representing the thread tag
   * \param IterVar iv The thread index to be binded;
   */
  virtual void BindThreadIndex(const IterVar& iv);                             // NOLINT(*)
  virtual void PrintStorageScope(const std::string& scope, std::ostream& os);  // NOLINT(*)
  virtual void PrintStorageSync(const CallNode* op);                           // NOLINT(*)
  /*! \brief reserves common Python keywords */
  void ReserveKeywordsAsUnique();

private:
  /*! \brief set of volatile buf access */
  std::unordered_set<const VarNode*> volatile_buf_;
  // deep comparison of PrimExpr
  ExprDeepEqual deep_equal_;
  // binding of let variables. Enables duplicate var defs that map to same value
  std::unordered_map<Var, const LetNode*, ObjectPtrHash, ObjectPtrEqual> let_binding_;
  /*! \brief Map of parameter to buffer */
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> var_buf_map_;
};



}  // namespace codegen
}  // namespace tvm

#endif