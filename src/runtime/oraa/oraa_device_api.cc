/*!
 * \file oraa_device_api.cc
 * \brief ORAA specific API
 */
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include <cstring>

namespace tvm {
namespace runtime {

class ORAADeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final {}
  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) final {}
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hine) final {
    return nullptr;
  }
  void FreeDataSpace(Device dev, void* ptr) final {}

 protected:
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {}

 public:
  TVMStreamHandle CreateStream(Device dev) { return nullptr; }
  void StreamSync(Device dev, TVMStreamHandle stream) final {}
  static ORAADeviceAPI* Global() {
    static auto* inst = new ORAADeviceAPI();
    return inst;
  }
};

TVM_REGISTER_GLOBAL("device_api.oraa").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = ORAADeviceAPI::Global();
  *rv = static_cast<void*>(ptr);
});

}  // namespace runtime
}  // namespace tvm