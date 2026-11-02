# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
@c.record
class QnnInterface_t(c.Struct):
  SIZE = 608
  backendId: int
  providerName: c.POINTER[ctypes.c_char]
  apiVersion: Qnn_ApiVersion_t
  v2_32: QnnInterface_ImplementationV2_32_t
uint32_t: TypeAlias = ctypes.c_uint32
@c.record
class Qnn_ApiVersion_t(c.Struct):
  SIZE = 24
  coreApiVersion: Qnn_Version_t
  backendApiVersion: Qnn_Version_t
@c.record
class Qnn_Version_t(c.Struct):
  SIZE = 12
  major: int
  minor: int
  patch: int
Qnn_Version_t.register_fields([('major', uint32_t, 0), ('minor', uint32_t, 4), ('patch', uint32_t, 8)])
Qnn_ApiVersion_t.register_fields([('coreApiVersion', Qnn_Version_t, 0), ('backendApiVersion', Qnn_Version_t, 12)])
@c.record
class QnnInterface_ImplementationV2_32_t(c.Struct):
  SIZE = 568
  propertyHasCapability: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_uint32]]
  backendCreate: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnBackend_Config_t]], c.POINTER[ctypes.c_void_p]]]
  backendSetConfig: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnBackend_Config_t]]]]
  backendGetApiVersion: c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[Qnn_ApiVersion_t]]]
  backendGetBuildId: c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[c.POINTER[ctypes.c_char]]]]
  backendRegisterOpPackage: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[ctypes.c_char], c.POINTER[ctypes.c_char], c.POINTER[ctypes.c_char]]]
  backendGetSupportedOperations: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[ctypes.c_uint32], c.POINTER[c.POINTER[QnnBackend_OperationName_t]]]]
  backendValidateOpConfig: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, Qnn_OpConfig_t]]
  backendFree: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p]]
  contextCreate: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, c.POINTER[c.POINTER[QnnContext_Config_t]], c.POINTER[ctypes.c_void_p]]]
  contextSetConfig: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnContext_Config_t]]]]
  contextGetBinarySize: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[ctypes.c_uint64]]]
  contextGetBinary: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, c.POINTER[ctypes.c_uint64]]]
  contextCreateFromBinary: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, c.POINTER[c.POINTER[QnnContext_Config_t]], ctypes.c_void_p, ctypes.c_uint64, c.POINTER[ctypes.c_void_p], ctypes.c_void_p]]
  contextFree: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p]]
  graphCreate: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[ctypes.c_char], c.POINTER[c.POINTER[QnnGraph_Config_t]], c.POINTER[ctypes.c_void_p]]]
  graphCreateSubgraph: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[ctypes.c_char], c.POINTER[ctypes.c_void_p]]]
  graphSetConfig: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnGraph_Config_t]]]]
  graphAddNode: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, Qnn_OpConfig_t]]
  graphFinalize: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]]
  graphRetrieve: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[ctypes.c_char], c.POINTER[ctypes.c_void_p]]]
  graphExecute: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[Qnn_Tensor_t], ctypes.c_uint32, c.POINTER[Qnn_Tensor_t], ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]]
  graphExecuteAsync: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[Qnn_Tensor_t], ctypes.c_uint32, c.POINTER[Qnn_Tensor_t], ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, c.CFUNCTYPE[None, [ctypes.c_void_p, Qnn_NotifyStatus_t]], ctypes.c_void_p]]
  tensorCreateContextTensor: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[Qnn_Tensor_t]]]
  tensorCreateGraphTensor: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[Qnn_Tensor_t]]]
  logCreate: c.CFUNCTYPE[ctypes.c_uint64, [c.CFUNCTYPE[None, [c.POINTER[ctypes.c_char], ctypes.c_uint32, ctypes.c_uint64, c.POINTER[struct___va_list_tag]]], ctypes.c_uint32, c.POINTER[ctypes.c_void_p]]]
  logSetLogLevel: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_uint32]]
  logFree: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p]]
  profileCreate: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_uint32, c.POINTER[ctypes.c_void_p]]]
  profileSetConfig: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnProfile_Config_t]]]]
  profileGetEvents: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[ctypes.c_uint64]], c.POINTER[ctypes.c_uint32]]]
  profileGetSubEvents: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_uint64, c.POINTER[c.POINTER[ctypes.c_uint64]], c.POINTER[ctypes.c_uint32]]]
  profileGetEventData: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_uint64, c.POINTER[QnnProfile_EventData_t]]]
  profileGetExtendedEventData: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_uint64, c.POINTER[QnnProfile_ExtendedEventData_t]]]
  profileFree: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p]]
  memRegister: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[Qnn_MemDescriptor_t], ctypes.c_uint32, c.POINTER[ctypes.c_void_p]]]
  memDeRegister: c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[ctypes.c_void_p], ctypes.c_uint32]]
  deviceGetPlatformInfo: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnDevice_PlatformInfo_t]]]]
  deviceFreePlatformInfo: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[QnnDevice_PlatformInfo_t]]]
  deviceGetInfrastructure: c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[c.POINTER[struct__QnnDevice_Infrastructure_t]]]]
  deviceCreate: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnDevice_Config_t]], c.POINTER[ctypes.c_void_p]]]
  deviceSetConfig: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnDevice_Config_t]]]]
  deviceGetInfo: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnDevice_PlatformInfo_t]]]]
  deviceFree: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p]]
  signalCreate: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnSignal_Config_t]], c.POINTER[ctypes.c_void_p]]]
  signalSetConfig: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnSignal_Config_t]]]]
  signalTrigger: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p]]
  signalFree: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p]]
  errorGetMessage: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_uint64, c.POINTER[c.POINTER[ctypes.c_char]]]]
  errorGetVerboseMessage: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_uint64, c.POINTER[c.POINTER[ctypes.c_char]]]]
  errorFreeVerboseMessage: c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[ctypes.c_char]]]
  graphPrepareExecutionEnvironment: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnGraph_ExecuteEnvironment_t]], ctypes.c_uint32]]
  graphReleaseExecutionEnvironment: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnGraph_ExecuteEnvironment_t]], ctypes.c_uint32]]
  graphGetProperty: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnGraph_Property_t]]]]
  contextValidateBinary: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, c.POINTER[c.POINTER[QnnContext_Config_t]], ctypes.c_void_p, ctypes.c_uint64]]
  contextCreateFromBinaryWithSignal: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, c.POINTER[c.POINTER[QnnContext_Config_t]], ctypes.c_void_p, ctypes.c_uint64, c.POINTER[ctypes.c_void_p], ctypes.c_void_p, ctypes.c_void_p]]
  contextCreateFromBinaryListAsync: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, c.POINTER[c.POINTER[QnnContext_Params_t]], c.POINTER[c.POINTER[QnnContext_Config_t]], ctypes.c_void_p]]
  tensorUpdateGraphTensors: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[Qnn_Tensor_t]], ctypes.c_uint64]]
  tensorUpdateContextTensors: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[Qnn_Tensor_t]], ctypes.c_uint64]]
  contextGetBinarySectionSize: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, c.POINTER[ctypes.c_uint64]]]
  contextGetBinarySection: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, c.POINTER[QnnContext_Buffer_t], c.POINTER[ctypes.c_uint64], ctypes.c_void_p, ctypes.c_void_p]]
  contextApplyBinarySection: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, c.POINTER[QnnContext_Buffer_t], ctypes.c_void_p, ctypes.c_void_p]]
  backendGetProperty: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnBackend_Property_t]]]]
  contextGetProperty: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnContext_Property_t]]]]
  contextGetIncrementalBinary: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[ctypes.c_void_p], c.POINTER[ctypes.c_uint64], c.POINTER[ctypes.c_uint64]]]
  contextReleaseIncrementalBinary: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]]
  contextFinalize: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p]]
  globalConfigSet: c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[c.POINTER[QnnGlobalConfig_t]]]]
  contextCreateFromBinaryWithCallback: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, c.POINTER[c.POINTER[QnnContext_Config_t]], c.POINTER[Qnn_ContextBinaryCallback_t], ctypes.c_void_p, ctypes.c_uint64, c.POINTER[ctypes.c_void_p], ctypes.c_void_p, ctypes.c_void_p]]
  contextGetBinarySectionUpdate: c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[QnnContext_Buffer_t], c.POINTER[QnnContext_Buffer_t], c.POINTER[c.POINTER[Qnn_Tensor_t]], ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c.POINTER[QnnContext_Buffer_t]]]
  contextFreeBinarySectionUpdate: c.CFUNCTYPE[ctypes.c_uint64, [QnnContext_Buffer_t, ctypes.c_void_p]]
QnnProperty_HasCapabilityFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_uint32]]
@c.record
class QnnBackend_Config_t(c.Struct):
  SIZE = 16
  option: int
  customConfig: ctypes.c_void_p
  errorConfig: Qnn_ErrorReportingConfig_t
  platformOption: c.POINTER[ctypes.c_char]
QnnBackend_ConfigOption_t: dict[int, str] = {(QNN_BACKEND_CONFIG_OPTION_CUSTOM:=0): 'QNN_BACKEND_CONFIG_OPTION_CUSTOM', (QNN_BACKEND_CONFIG_OPTION_ERROR_REPORTING:=1): 'QNN_BACKEND_CONFIG_OPTION_ERROR_REPORTING', (QNN_BACKEND_CONFIG_OPTION_PLATFORM:=2): 'QNN_BACKEND_CONFIG_OPTION_PLATFORM', (QNN_BACKEND_CONFIG_OPTION_UNDEFINED:=2147483647): 'QNN_BACKEND_CONFIG_OPTION_UNDEFINED'}
QnnBackend_CustomConfig_t: TypeAlias = ctypes.c_void_p
@c.record
class Qnn_ErrorReportingConfig_t(c.Struct):
  SIZE = 8
  reportingLevel: int
  storageLimit: int
Qnn_ErrorReportingLevel_t: dict[int, str] = {(QNN_ERROR_REPORTING_LEVEL_BRIEF:=0): 'QNN_ERROR_REPORTING_LEVEL_BRIEF', (QNN_ERROR_REPORTING_LEVEL_DETAILED:=1): 'QNN_ERROR_REPORTING_LEVEL_DETAILED', (QNN_ERROR_REPORTING_LEVEL_UNDEFINED:=2147483647): 'QNN_ERROR_REPORTING_LEVEL_UNDEFINED'}
Qnn_ErrorReportingConfig_t.register_fields([('reportingLevel', ctypes.c_uint32, 0), ('storageLimit', uint32_t, 4)])
QnnBackend_Config_t.register_fields([('option', ctypes.c_uint32, 0), ('customConfig', QnnBackend_CustomConfig_t, 8), ('errorConfig', Qnn_ErrorReportingConfig_t, 8), ('platformOption', c.POINTER[ctypes.c_char], 8)])
QnnBackend_CreateFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnBackend_Config_t]], c.POINTER[ctypes.c_void_p]]]
QnnBackend_SetConfigFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnBackend_Config_t]]]]
QnnBackend_GetApiVersionFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[Qnn_ApiVersion_t]]]
QnnBackend_GetBuildIdFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[c.POINTER[ctypes.c_char]]]]
QnnBackend_RegisterOpPackageFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[ctypes.c_char], c.POINTER[ctypes.c_char], c.POINTER[ctypes.c_char]]]
@c.record
class QnnBackend_OperationName_t(c.Struct):
  SIZE = 24
  packageName: c.POINTER[ctypes.c_char]
  name: c.POINTER[ctypes.c_char]
  target: c.POINTER[ctypes.c_char]
QnnBackend_OperationName_t.register_fields([('packageName', c.POINTER[ctypes.c_char], 0), ('name', c.POINTER[ctypes.c_char], 8), ('target', c.POINTER[ctypes.c_char], 16)])
QnnBackend_GetSupportedOperationsFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[ctypes.c_uint32], c.POINTER[c.POINTER[QnnBackend_OperationName_t]]]]
@c.record
class Qnn_OpConfig_t(c.Struct):
  SIZE = 80
  version: int
  v1: Qnn_OpConfigV1_t
Qnn_OpConfigVersion_t: dict[int, str] = {(QNN_OPCONFIG_VERSION_1:=1): 'QNN_OPCONFIG_VERSION_1', (QNN_OPCONFIG_VERSION_UNDEFINED:=2147483647): 'QNN_OPCONFIG_VERSION_UNDEFINED'}
@c.record
class Qnn_OpConfigV1_t(c.Struct):
  SIZE = 72
  name: c.POINTER[ctypes.c_char]
  packageName: c.POINTER[ctypes.c_char]
  typeName: c.POINTER[ctypes.c_char]
  numOfParams: int
  params: c.POINTER[Qnn_Param_t]
  numOfInputs: int
  inputTensors: c.POINTER[Qnn_Tensor_t]
  numOfOutputs: int
  outputTensors: c.POINTER[Qnn_Tensor_t]
@c.record
class Qnn_Param_t(c.Struct):
  SIZE = 160
  paramType: int
  name: c.POINTER[ctypes.c_char]
  scalarParam: Qnn_Scalar_t
  tensorParam: Qnn_Tensor_t
Qnn_ParamType_t: dict[int, str] = {(QNN_PARAMTYPE_SCALAR:=0): 'QNN_PARAMTYPE_SCALAR', (QNN_PARAMTYPE_TENSOR:=1): 'QNN_PARAMTYPE_TENSOR', (QNN_PARAMTYPE_UNDEFINED:=4294967295): 'QNN_PARAMTYPE_UNDEFINED'}
@c.record
class Qnn_Scalar_t(c.Struct):
  SIZE = 16
  dataType: int
  floatValue: float
  doubleValue: float
  uint64Value: int
  int64Value: int
  uint32Value: int
  int32Value: int
  uint16Value: int
  int16Value: int
  uint8Value: int
  int8Value: int
  bool8Value: int
  bfloat16Value: int
  stringValue: c.POINTER[ctypes.c_char]
Qnn_DataType_t: dict[int, str] = {(QNN_DATATYPE_INT_8:=8): 'QNN_DATATYPE_INT_8', (QNN_DATATYPE_INT_16:=22): 'QNN_DATATYPE_INT_16', (QNN_DATATYPE_INT_32:=50): 'QNN_DATATYPE_INT_32', (QNN_DATATYPE_INT_64:=100): 'QNN_DATATYPE_INT_64', (QNN_DATATYPE_UINT_8:=264): 'QNN_DATATYPE_UINT_8', (QNN_DATATYPE_UINT_16:=278): 'QNN_DATATYPE_UINT_16', (QNN_DATATYPE_UINT_32:=306): 'QNN_DATATYPE_UINT_32', (QNN_DATATYPE_UINT_64:=356): 'QNN_DATATYPE_UINT_64', (QNN_DATATYPE_FLOAT_16:=534): 'QNN_DATATYPE_FLOAT_16', (QNN_DATATYPE_BFLOAT_16:=550): 'QNN_DATATYPE_BFLOAT_16', (QNN_DATATYPE_FLOAT_32:=562): 'QNN_DATATYPE_FLOAT_32', (QNN_DATATYPE_FLOAT_64:=612): 'QNN_DATATYPE_FLOAT_64', (QNN_DATATYPE_SFIXED_POINT_4:=772): 'QNN_DATATYPE_SFIXED_POINT_4', (QNN_DATATYPE_SFIXED_POINT_8:=776): 'QNN_DATATYPE_SFIXED_POINT_8', (QNN_DATATYPE_SFIXED_POINT_16:=790): 'QNN_DATATYPE_SFIXED_POINT_16', (QNN_DATATYPE_SFIXED_POINT_32:=818): 'QNN_DATATYPE_SFIXED_POINT_32', (QNN_DATATYPE_UFIXED_POINT_4:=1028): 'QNN_DATATYPE_UFIXED_POINT_4', (QNN_DATATYPE_UFIXED_POINT_8:=1032): 'QNN_DATATYPE_UFIXED_POINT_8', (QNN_DATATYPE_UFIXED_POINT_16:=1046): 'QNN_DATATYPE_UFIXED_POINT_16', (QNN_DATATYPE_UFIXED_POINT_32:=1074): 'QNN_DATATYPE_UFIXED_POINT_32', (QNN_DATATYPE_BOOL_8:=1288): 'QNN_DATATYPE_BOOL_8', (QNN_DATATYPE_STRING:=1544): 'QNN_DATATYPE_STRING', (QNN_DATATYPE_UNDEFINED:=2147483647): 'QNN_DATATYPE_UNDEFINED'}
uint64_t: TypeAlias = ctypes.c_uint64
int64_t: TypeAlias = ctypes.c_int64
int32_t: TypeAlias = ctypes.c_int32
uint16_t: TypeAlias = ctypes.c_uint16
int16_t: TypeAlias = ctypes.c_int16
uint8_t: TypeAlias = ctypes.c_ubyte
int8_t: TypeAlias = ctypes.c_byte
Qnn_Scalar_t.register_fields([('dataType', ctypes.c_uint32, 0), ('floatValue', ctypes.c_float, 8), ('doubleValue', ctypes.c_double, 8), ('uint64Value', uint64_t, 8), ('int64Value', int64_t, 8), ('uint32Value', uint32_t, 8), ('int32Value', int32_t, 8), ('uint16Value', uint16_t, 8), ('int16Value', int16_t, 8), ('uint8Value', uint8_t, 8), ('int8Value', int8_t, 8), ('bool8Value', uint8_t, 8), ('bfloat16Value', int16_t, 8), ('stringValue', c.POINTER[ctypes.c_char], 8)])
@c.record
class Qnn_Tensor_t(c.Struct):
  SIZE = 144
  version: int
  v1: Qnn_TensorV1_t
  v2: Qnn_TensorV2_t
Qnn_TensorVersion_t: dict[int, str] = {(QNN_TENSOR_VERSION_1:=1): 'QNN_TENSOR_VERSION_1', (QNN_TENSOR_VERSION_2:=2): 'QNN_TENSOR_VERSION_2', (QNN_TENSOR_VERSION_UNDEFINED:=2147483647): 'QNN_TENSOR_VERSION_UNDEFINED'}
@c.record
class Qnn_TensorV1_t(c.Struct):
  SIZE = 112
  id: int
  name: c.POINTER[ctypes.c_char]
  type: int
  dataFormat: int
  dataType: int
  quantizeParams: Qnn_QuantizeParams_t
  rank: int
  dimensions: c.POINTER[ctypes.c_uint32]
  memType: int
  clientBuf: Qnn_ClientBuffer_t
  memHandle: ctypes.c_void_p
Qnn_TensorType_t: dict[int, str] = {(QNN_TENSOR_TYPE_APP_WRITE:=0): 'QNN_TENSOR_TYPE_APP_WRITE', (QNN_TENSOR_TYPE_APP_READ:=1): 'QNN_TENSOR_TYPE_APP_READ', (QNN_TENSOR_TYPE_APP_READWRITE:=2): 'QNN_TENSOR_TYPE_APP_READWRITE', (QNN_TENSOR_TYPE_NATIVE:=3): 'QNN_TENSOR_TYPE_NATIVE', (QNN_TENSOR_TYPE_STATIC:=4): 'QNN_TENSOR_TYPE_STATIC', (QNN_TENSOR_TYPE_NULL:=5): 'QNN_TENSOR_TYPE_NULL', (QNN_TENSOR_TYPE_UPDATEABLE_STATIC:=6): 'QNN_TENSOR_TYPE_UPDATEABLE_STATIC', (QNN_TENSOR_TYPE_UPDATEABLE_NATIVE:=7): 'QNN_TENSOR_TYPE_UPDATEABLE_NATIVE', (QNN_TENSOR_TYPE_UPDATEABLE_APP_WRITE:=8): 'QNN_TENSOR_TYPE_UPDATEABLE_APP_WRITE', (QNN_TENSOR_TYPE_UPDATEABLE_APP_READ:=9): 'QNN_TENSOR_TYPE_UPDATEABLE_APP_READ', (QNN_TENSOR_TYPE_UPDATEABLE_APP_READWRITE:=10): 'QNN_TENSOR_TYPE_UPDATEABLE_APP_READWRITE', (QNN_TENSOR_TYPE_OPTIONAL_APP_WRITE:=11): 'QNN_TENSOR_TYPE_OPTIONAL_APP_WRITE', (QNN_TENSOR_TYPE_OPTIONAL_APP_READ:=12): 'QNN_TENSOR_TYPE_OPTIONAL_APP_READ', (QNN_TENSOR_TYPE_OPTIONAL_APP_READWRITE:=13): 'QNN_TENSOR_TYPE_OPTIONAL_APP_READWRITE', (QNN_TENSOR_TYPE_UNDEFINED:=2147483647): 'QNN_TENSOR_TYPE_UNDEFINED'}
Qnn_TensorDataFormat_t: TypeAlias = ctypes.c_uint32
@c.record
class Qnn_QuantizeParams_t(c.Struct):
  SIZE = 40
  encodingDefinition: int
  quantizationEncoding: int
  scaleOffsetEncoding: Qnn_ScaleOffset_t
  axisScaleOffsetEncoding: Qnn_AxisScaleOffset_t
  bwScaleOffsetEncoding: Qnn_BwScaleOffset_t
  bwAxisScaleOffsetEncoding: Qnn_BwAxisScaleOffset_t
  blockEncoding: Qnn_BlockEncoding_t
  blockwiseExpansion: c.POINTER[Qnn_BlockwiseExpansion_t]
  vectorEncoding: c.POINTER[Qnn_VectorEncoding_t]
  floatBlockEncoding: Qnn_FloatBlockEncoding_t
  bwAxisScaleOffsetMappedEncoding: c.POINTER[Qnn_BwAxisScaleOffsetMapped_t]
  bwBlockMappedEncoding: c.POINTER[Qnn_BwBlockMapped_t]
  bwBlockwiseExpansionMappedEncoding: c.POINTER[Qnn_BwBlockwiseExpansionMapped_t]
  bwFloatBlockEncoding: Qnn_BwFloatBlockEncoding_t
Qnn_Definition_t: dict[int, str] = {(QNN_DEFINITION_IMPL_GENERATED:=0): 'QNN_DEFINITION_IMPL_GENERATED', (QNN_DEFINITION_DEFINED:=1): 'QNN_DEFINITION_DEFINED', (QNN_DEFINITION_UNDEFINED:=2147483647): 'QNN_DEFINITION_UNDEFINED'}
Qnn_QuantizationEncoding_t: dict[int, str] = {(QNN_QUANTIZATION_ENCODING_SCALE_OFFSET:=0): 'QNN_QUANTIZATION_ENCODING_SCALE_OFFSET', (QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET:=1): 'QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET', (QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET:=2): 'QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET', (QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET:=3): 'QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET', (QNN_QUANTIZATION_ENCODING_BLOCK:=4): 'QNN_QUANTIZATION_ENCODING_BLOCK', (QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION:=5): 'QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION', (QNN_QUANTIZATION_ENCODING_VECTOR:=6): 'QNN_QUANTIZATION_ENCODING_VECTOR', (QNN_QUANTIZATION_ENCODING_FLOAT_BLOCK:=7): 'QNN_QUANTIZATION_ENCODING_FLOAT_BLOCK', (QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET_MAPPED:=8): 'QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET_MAPPED', (QNN_QUANTIZATION_ENCODING_BW_BLOCK_MAPPED:=9): 'QNN_QUANTIZATION_ENCODING_BW_BLOCK_MAPPED', (QNN_QUANTIZATION_ENCODING_BW_BLOCKWISE_EXPANSION_MAPPED:=10): 'QNN_QUANTIZATION_ENCODING_BW_BLOCKWISE_EXPANSION_MAPPED', (QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK:=11): 'QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK', (QNN_QUANTIZATION_ENCODING_UNDEFINED:=2147483647): 'QNN_QUANTIZATION_ENCODING_UNDEFINED'}
@c.record
class Qnn_ScaleOffset_t(c.Struct):
  SIZE = 8
  scale: float
  offset: int
Qnn_ScaleOffset_t.register_fields([('scale', ctypes.c_float, 0), ('offset', int32_t, 4)])
@c.record
class Qnn_AxisScaleOffset_t(c.Struct):
  SIZE = 16
  axis: int
  numScaleOffsets: int
  scaleOffset: c.POINTER[Qnn_ScaleOffset_t]
Qnn_AxisScaleOffset_t.register_fields([('axis', int32_t, 0), ('numScaleOffsets', uint32_t, 4), ('scaleOffset', c.POINTER[Qnn_ScaleOffset_t], 8)])
@c.record
class Qnn_BwScaleOffset_t(c.Struct):
  SIZE = 12
  bitwidth: int
  scale: float
  offset: int
Qnn_BwScaleOffset_t.register_fields([('bitwidth', uint32_t, 0), ('scale', ctypes.c_float, 4), ('offset', int32_t, 8)])
@c.record
class Qnn_BwAxisScaleOffset_t(c.Struct):
  SIZE = 32
  bitwidth: int
  axis: int
  numElements: int
  scales: c.POINTER[ctypes.c_float]
  offsets: c.POINTER[ctypes.c_int32]
Qnn_BwAxisScaleOffset_t.register_fields([('bitwidth', uint32_t, 0), ('axis', int32_t, 4), ('numElements', uint32_t, 8), ('scales', c.POINTER[ctypes.c_float], 16), ('offsets', c.POINTER[int32_t], 24)])
@c.record
class Qnn_BlockEncoding_t(c.Struct):
  SIZE = 16
  blockSize: c.POINTER[ctypes.c_uint32]
  scaleOffset: c.POINTER[Qnn_ScaleOffset_t]
Qnn_BlockEncoding_t.register_fields([('blockSize', c.POINTER[uint32_t], 0), ('scaleOffset', c.POINTER[Qnn_ScaleOffset_t], 8)])
@c.record
class Qnn_BlockwiseExpansion_t(c.Struct):
  SIZE = 40
  axis: int
  scaleOffsets: c.POINTER[Qnn_ScaleOffset_t]
  numBlocksPerAxis: int
  blockScaleBitwidth: int
  blockScaleStorageType: int
  blocksScale8: c.POINTER[ctypes.c_ubyte]
  blocksScale16: c.POINTER[ctypes.c_uint16]
Qnn_BlockwiseExpansionBlockScaleStorageType_t: dict[int, str] = {(QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_8:=0): 'QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_8', (QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_16:=1): 'QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_16', (QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_UNDEFINED:=2147483647): 'QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_UNDEFINED'}
Qnn_BlockwiseExpansion_t.register_fields([('axis', int32_t, 0), ('scaleOffsets', c.POINTER[Qnn_ScaleOffset_t], 8), ('numBlocksPerAxis', uint32_t, 16), ('blockScaleBitwidth', uint32_t, 20), ('blockScaleStorageType', ctypes.c_uint32, 24), ('blocksScale8', c.POINTER[uint8_t], 32), ('blocksScale16', c.POINTER[uint16_t], 32)])
@c.record
class Qnn_VectorEncoding_t(c.Struct):
  SIZE = 48
  bwAxisScaleOffset: Qnn_BwAxisScaleOffset_t
  rowsPerBlock: int
  columnsPerBlock: int
  vectorDimension: int
  vectorStride: int
  indexBitwidth: int
Qnn_VectorEncoding_t.register_fields([('bwAxisScaleOffset', Qnn_BwAxisScaleOffset_t, 0), ('rowsPerBlock', uint32_t, 32), ('columnsPerBlock', uint32_t, 36), ('vectorDimension', uint8_t, 40), ('vectorStride', uint8_t, 41), ('indexBitwidth', uint8_t, 42)])
@c.record
class Qnn_FloatBlockEncoding_t(c.Struct):
  SIZE = 16
  blockSize: c.POINTER[ctypes.c_uint32]
  floatScaleOffset: c.POINTER[Qnn_FloatScaleOffset_t]
@c.record
class Qnn_FloatScaleOffset_t(c.Struct):
  SIZE = 8
  scale: float
  offset: float
Qnn_FloatScaleOffset_t.register_fields([('scale', ctypes.c_float, 0), ('offset', ctypes.c_float, 4)])
Qnn_FloatBlockEncoding_t.register_fields([('blockSize', c.POINTER[uint32_t], 0), ('floatScaleOffset', c.POINTER[Qnn_FloatScaleOffset_t], 8)])
@c.record
class Qnn_BwAxisScaleOffsetMapped_t(c.Struct):
  SIZE = 32
  bitwidth: int
  axis: int
  mapping: int
  numElements: int
  scales: c.POINTER[ctypes.c_float]
  offsets: c.POINTER[ctypes.c_int32]
Qnn_QuantizationEncodingMapping_t: dict[int, str] = {(QNN_QUANTIZATION_ENCODING_MAPPING_STANDARD_SYMMETRIC:=0): 'QNN_QUANTIZATION_ENCODING_MAPPING_STANDARD_SYMMETRIC', (QNN_QUANTIZATION_ENCODING_MAPPING_ASYMMETRIC_PLUS_ONE:=1): 'QNN_QUANTIZATION_ENCODING_MAPPING_ASYMMETRIC_PLUS_ONE', (QNN_QUANTIZATION_ENCODING_MAPPING_LINEAR_SYMMETRIC_EXCLUDE_ZERO:=2): 'QNN_QUANTIZATION_ENCODING_MAPPING_LINEAR_SYMMETRIC_EXCLUDE_ZERO', (QNN_QUANTIZATION_ENCODING_MAPPING_UNDEFINED:=2147483647): 'QNN_QUANTIZATION_ENCODING_MAPPING_UNDEFINED'}
Qnn_BwAxisScaleOffsetMapped_t.register_fields([('bitwidth', uint32_t, 0), ('axis', int32_t, 4), ('mapping', ctypes.c_uint32, 8), ('numElements', uint32_t, 12), ('scales', c.POINTER[ctypes.c_float], 16), ('offsets', c.POINTER[int32_t], 24)])
@c.record
class Qnn_BwBlockMapped_t(c.Struct):
  SIZE = 24
  bitwidth: int
  mapping: int
  blockSize: c.POINTER[ctypes.c_uint32]
  scaleOffset: c.POINTER[Qnn_ScaleOffset_t]
Qnn_BwBlockMapped_t.register_fields([('bitwidth', uint32_t, 0), ('mapping', ctypes.c_uint32, 4), ('blockSize', c.POINTER[uint32_t], 8), ('scaleOffset', c.POINTER[Qnn_ScaleOffset_t], 16)])
@c.record
class Qnn_BwBlockwiseExpansionMapped_t(c.Struct):
  SIZE = 48
  bitwidth: int
  mapping: int
  axis: int
  scaleOffsets: c.POINTER[Qnn_ScaleOffset_t]
  numBlocksPerAxis: int
  blockScaleBitwidth: int
  blockScaleStorageType: int
  blocksScale8: c.POINTER[ctypes.c_ubyte]
  blocksScale16: c.POINTER[ctypes.c_uint16]
Qnn_BwBlockwiseExpansionMapped_t.register_fields([('bitwidth', uint32_t, 0), ('mapping', ctypes.c_uint32, 4), ('axis', int32_t, 8), ('scaleOffsets', c.POINTER[Qnn_ScaleOffset_t], 16), ('numBlocksPerAxis', uint32_t, 24), ('blockScaleBitwidth', uint32_t, 28), ('blockScaleStorageType', ctypes.c_uint32, 32), ('blocksScale8', c.POINTER[uint8_t], 40), ('blocksScale16', c.POINTER[uint16_t], 40)])
@c.record
class Qnn_BwFloatBlockEncoding_t(c.Struct):
  SIZE = 24
  bitwidth: int
  blockSize: c.POINTER[ctypes.c_uint32]
  floatScaleOffset: c.POINTER[Qnn_FloatScaleOffset_t]
Qnn_BwFloatBlockEncoding_t.register_fields([('bitwidth', uint32_t, 0), ('blockSize', c.POINTER[uint32_t], 8), ('floatScaleOffset', c.POINTER[Qnn_FloatScaleOffset_t], 16)])
Qnn_QuantizeParams_t.register_fields([('encodingDefinition', ctypes.c_uint32, 0), ('quantizationEncoding', ctypes.c_uint32, 4), ('scaleOffsetEncoding', Qnn_ScaleOffset_t, 8), ('axisScaleOffsetEncoding', Qnn_AxisScaleOffset_t, 8), ('bwScaleOffsetEncoding', Qnn_BwScaleOffset_t, 8), ('bwAxisScaleOffsetEncoding', Qnn_BwAxisScaleOffset_t, 8), ('blockEncoding', Qnn_BlockEncoding_t, 8), ('blockwiseExpansion', c.POINTER[Qnn_BlockwiseExpansion_t], 8), ('vectorEncoding', c.POINTER[Qnn_VectorEncoding_t], 8), ('floatBlockEncoding', Qnn_FloatBlockEncoding_t, 8), ('bwAxisScaleOffsetMappedEncoding', c.POINTER[Qnn_BwAxisScaleOffsetMapped_t], 8), ('bwBlockMappedEncoding', c.POINTER[Qnn_BwBlockMapped_t], 8), ('bwBlockwiseExpansionMappedEncoding', c.POINTER[Qnn_BwBlockwiseExpansionMapped_t], 8), ('bwFloatBlockEncoding', Qnn_BwFloatBlockEncoding_t, 8)])
Qnn_TensorMemType_t: dict[int, str] = {(QNN_TENSORMEMTYPE_RAW:=0): 'QNN_TENSORMEMTYPE_RAW', (QNN_TENSORMEMTYPE_MEMHANDLE:=1): 'QNN_TENSORMEMTYPE_MEMHANDLE', (QNN_TENSORMEMTYPE_RETRIEVE_RAW:=2): 'QNN_TENSORMEMTYPE_RETRIEVE_RAW', (QNN_TENSORMEMTYPE_UNDEFINED:=2147483647): 'QNN_TENSORMEMTYPE_UNDEFINED'}
@c.record
class Qnn_ClientBuffer_t(c.Struct):
  SIZE = 16
  data: ctypes.c_void_p
  dataSize: int
Qnn_ClientBuffer_t.register_fields([('data', ctypes.c_void_p, 0), ('dataSize', uint32_t, 8)])
Qnn_MemHandle_t: TypeAlias = ctypes.c_void_p
Qnn_TensorV1_t.register_fields([('id', uint32_t, 0), ('name', c.POINTER[ctypes.c_char], 8), ('type', ctypes.c_uint32, 16), ('dataFormat', Qnn_TensorDataFormat_t, 20), ('dataType', ctypes.c_uint32, 24), ('quantizeParams', Qnn_QuantizeParams_t, 32), ('rank', uint32_t, 72), ('dimensions', c.POINTER[uint32_t], 80), ('memType', ctypes.c_uint32, 88), ('clientBuf', Qnn_ClientBuffer_t, 96), ('memHandle', Qnn_MemHandle_t, 96)])
@c.record
class Qnn_TensorV2_t(c.Struct):
  SIZE = 136
  id: int
  name: c.POINTER[ctypes.c_char]
  type: int
  dataFormat: int
  dataType: int
  quantizeParams: Qnn_QuantizeParams_t
  rank: int
  dimensions: c.POINTER[ctypes.c_uint32]
  memType: int
  clientBuf: Qnn_ClientBuffer_t
  memHandle: ctypes.c_void_p
  retrieveRaw: c.POINTER[Qnn_TensorRetrieveRaw_t]
  isDynamicDimensions: c.POINTER[ctypes.c_ubyte]
  sparseParams: Qnn_SparseParams_t
  isProduced: int
@c.record
class Qnn_TensorRetrieveRaw_t(c.Struct):
  SIZE = 16
  getTensorData: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, c.POINTER[Qnn_ClientBuffer_t]]]
  freeTensorData: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]]
Qnn_GetTensorRawDataFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, c.POINTER[Qnn_ClientBuffer_t]]]
Qnn_FreeTensorRawDataFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]]
Qnn_TensorRetrieveRaw_t.register_fields([('getTensorData', Qnn_GetTensorRawDataFn_t, 0), ('freeTensorData', Qnn_FreeTensorRawDataFn_t, 8)])
@c.record
class Qnn_SparseParams_t(c.Struct):
  SIZE = 12
  type: int
  hybridCoo: Qnn_SparseLayoutHybridCoo_t
Qnn_SparseLayoutType_t: dict[int, str] = {(QNN_SPARSE_LAYOUT_HYBRID_COO:=0): 'QNN_SPARSE_LAYOUT_HYBRID_COO', (QNN_SPARSE_LAYOUT_UNDEFINED:=2147483647): 'QNN_SPARSE_LAYOUT_UNDEFINED'}
@c.record
class Qnn_SparseLayoutHybridCoo_t(c.Struct):
  SIZE = 8
  numSpecifiedElements: int
  numSparseDimensions: int
Qnn_SparseLayoutHybridCoo_t.register_fields([('numSpecifiedElements', uint32_t, 0), ('numSparseDimensions', uint32_t, 4)])
Qnn_SparseParams_t.register_fields([('type', ctypes.c_uint32, 0), ('hybridCoo', Qnn_SparseLayoutHybridCoo_t, 4)])
Qnn_TensorV2_t.register_fields([('id', uint32_t, 0), ('name', c.POINTER[ctypes.c_char], 8), ('type', ctypes.c_uint32, 16), ('dataFormat', Qnn_TensorDataFormat_t, 20), ('dataType', ctypes.c_uint32, 24), ('quantizeParams', Qnn_QuantizeParams_t, 32), ('rank', uint32_t, 72), ('dimensions', c.POINTER[uint32_t], 80), ('memType', ctypes.c_uint32, 88), ('clientBuf', Qnn_ClientBuffer_t, 96), ('memHandle', Qnn_MemHandle_t, 96), ('retrieveRaw', c.POINTER[Qnn_TensorRetrieveRaw_t], 96), ('isDynamicDimensions', c.POINTER[uint8_t], 112), ('sparseParams', Qnn_SparseParams_t, 120), ('isProduced', uint8_t, 132)])
Qnn_Tensor_t.register_fields([('version', ctypes.c_uint32, 0), ('v1', Qnn_TensorV1_t, 8), ('v2', Qnn_TensorV2_t, 8)])
Qnn_Param_t.register_fields([('paramType', ctypes.c_uint32, 0), ('name', c.POINTER[ctypes.c_char], 8), ('scalarParam', Qnn_Scalar_t, 16), ('tensorParam', Qnn_Tensor_t, 16)])
Qnn_OpConfigV1_t.register_fields([('name', c.POINTER[ctypes.c_char], 0), ('packageName', c.POINTER[ctypes.c_char], 8), ('typeName', c.POINTER[ctypes.c_char], 16), ('numOfParams', uint32_t, 24), ('params', c.POINTER[Qnn_Param_t], 32), ('numOfInputs', uint32_t, 40), ('inputTensors', c.POINTER[Qnn_Tensor_t], 48), ('numOfOutputs', uint32_t, 56), ('outputTensors', c.POINTER[Qnn_Tensor_t], 64)])
Qnn_OpConfig_t.register_fields([('version', ctypes.c_uint32, 0), ('v1', Qnn_OpConfigV1_t, 8)])
QnnBackend_ValidateOpConfigFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, Qnn_OpConfig_t]]
QnnBackend_FreeFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p]]
@c.record
class QnnContext_Config_t(c.Struct):
  SIZE = 16
  option: int
  customConfig: ctypes.c_void_p
  priority: int
  errorConfig: Qnn_ErrorReportingConfig_t
  oemString: c.POINTER[ctypes.c_char]
  asyncExeQueueDepth: QnnContext_AsyncExecutionQueueDepth_t
  enableGraphs: c.POINTER[c.POINTER[ctypes.c_char]]
  memoryLimitHint: int
  isPersistentBinary: int
  binaryCompatibilityType: int
  isGraphInitDeferred: int
  graphRetentionOrder: c.POINTER[c.POINTER[ctypes.c_char]]
QnnContext_ConfigOption_t: dict[int, str] = {(QNN_CONTEXT_CONFIG_OPTION_CUSTOM:=0): 'QNN_CONTEXT_CONFIG_OPTION_CUSTOM', (QNN_CONTEXT_CONFIG_OPTION_PRIORITY:=1): 'QNN_CONTEXT_CONFIG_OPTION_PRIORITY', (QNN_CONTEXT_CONFIG_OPTION_ERROR_REPORTING:=2): 'QNN_CONTEXT_CONFIG_OPTION_ERROR_REPORTING', (QNN_CONTEXT_CONFIG_OPTION_OEM_STRING:=3): 'QNN_CONTEXT_CONFIG_OPTION_OEM_STRING', (QNN_CONTEXT_CONFIG_ASYNC_EXECUTION_QUEUE_DEPTH:=4): 'QNN_CONTEXT_CONFIG_ASYNC_EXECUTION_QUEUE_DEPTH', (QNN_CONTEXT_CONFIG_ENABLE_GRAPHS:=5): 'QNN_CONTEXT_CONFIG_ENABLE_GRAPHS', (QNN_CONTEXT_CONFIG_MEMORY_LIMIT_HINT:=6): 'QNN_CONTEXT_CONFIG_MEMORY_LIMIT_HINT', (QNN_CONTEXT_CONFIG_PERSISTENT_BINARY:=7): 'QNN_CONTEXT_CONFIG_PERSISTENT_BINARY', (QNN_CONTEXT_CONFIG_BINARY_COMPATIBILITY:=8): 'QNN_CONTEXT_CONFIG_BINARY_COMPATIBILITY', (QNN_CONTEXT_CONFIG_OPTION_DEFER_GRAPH_INIT:=9): 'QNN_CONTEXT_CONFIG_OPTION_DEFER_GRAPH_INIT', (QNN_CONTEXT_CONFIG_GRAPH_RETENTION_ORDER:=10): 'QNN_CONTEXT_CONFIG_GRAPH_RETENTION_ORDER', (QNN_CONTEXT_CONFIG_UNDEFINED:=2147483647): 'QNN_CONTEXT_CONFIG_UNDEFINED'}
QnnContext_CustomConfig_t: TypeAlias = ctypes.c_void_p
Qnn_Priority_t: dict[int, str] = {(QNN_PRIORITY_LOW:=0): 'QNN_PRIORITY_LOW', (QNN_PRIORITY_NORMAL:=100): 'QNN_PRIORITY_NORMAL', (QNN_PRIORITY_DEFAULT:=100): 'QNN_PRIORITY_DEFAULT', (QNN_PRIORITY_NORMAL_HIGH:=150): 'QNN_PRIORITY_NORMAL_HIGH', (QNN_PRIORITY_HIGH:=200): 'QNN_PRIORITY_HIGH', (QNN_PRIORITY_UNDEFINED:=2147483647): 'QNN_PRIORITY_UNDEFINED'}
@c.record
class QnnContext_AsyncExecutionQueueDepth_t(c.Struct):
  SIZE = 8
  type: int
  depth: int
QnnContext_AsyncExecutionQueueDepthType_t: dict[int, str] = {(QNN_CONTEXT_ASYNC_EXECUTION_QUEUE_DEPTH_TYPE_NUMERIC:=0): 'QNN_CONTEXT_ASYNC_EXECUTION_QUEUE_DEPTH_TYPE_NUMERIC', (QNN_CONTEXT_ASYNC_EXECUTION_QUEUE_DEPTH_TYPE_UNDEFINED:=134217727): 'QNN_CONTEXT_ASYNC_EXECUTION_QUEUE_DEPTH_TYPE_UNDEFINED'}
QnnContext_AsyncExecutionQueueDepth_t.register_fields([('type', ctypes.c_uint32, 0), ('depth', uint32_t, 4)])
QnnContext_BinaryCompatibilityType_t: dict[int, str] = {(QNN_CONTEXT_BINARY_COMPATIBILITY_PERMISSIVE:=0): 'QNN_CONTEXT_BINARY_COMPATIBILITY_PERMISSIVE', (QNN_CONTEXT_BINARY_COMPATIBILITY_STRICT:=1): 'QNN_CONTEXT_BINARY_COMPATIBILITY_STRICT', (QNN_CONTEXT_BINARY_COMPATIBILITY_TYPE_UNDEFINED:=134217727): 'QNN_CONTEXT_BINARY_COMPATIBILITY_TYPE_UNDEFINED'}
QnnContext_Config_t.register_fields([('option', ctypes.c_uint32, 0), ('customConfig', QnnContext_CustomConfig_t, 8), ('priority', ctypes.c_uint32, 8), ('errorConfig', Qnn_ErrorReportingConfig_t, 8), ('oemString', c.POINTER[ctypes.c_char], 8), ('asyncExeQueueDepth', QnnContext_AsyncExecutionQueueDepth_t, 8), ('enableGraphs', c.POINTER[c.POINTER[ctypes.c_char]], 8), ('memoryLimitHint', uint64_t, 8), ('isPersistentBinary', uint8_t, 8), ('binaryCompatibilityType', ctypes.c_uint32, 8), ('isGraphInitDeferred', uint8_t, 8), ('graphRetentionOrder', c.POINTER[c.POINTER[ctypes.c_char]], 8)])
QnnContext_CreateFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, c.POINTER[c.POINTER[QnnContext_Config_t]], c.POINTER[ctypes.c_void_p]]]
QnnContext_SetConfigFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnContext_Config_t]]]]
QnnContext_GetBinarySizeFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[ctypes.c_uint64]]]
QnnContext_GetBinaryFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, c.POINTER[ctypes.c_uint64]]]
QnnContext_CreateFromBinaryFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, c.POINTER[c.POINTER[QnnContext_Config_t]], ctypes.c_void_p, ctypes.c_uint64, c.POINTER[ctypes.c_void_p], ctypes.c_void_p]]
QnnContext_FreeFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p]]
@c.record
class QnnGraph_Config_t(c.Struct):
  SIZE = 16
  option: int
  customConfig: ctypes.c_void_p
  priority: int
  profileHandle: ctypes.c_void_p
  profilingState: int
  numProfilingExecutions: int
  enableBinarySectionWeightsUpdates: int
QnnGraph_ConfigOption_t: dict[int, str] = {(QNN_GRAPH_CONFIG_OPTION_CUSTOM:=0): 'QNN_GRAPH_CONFIG_OPTION_CUSTOM', (QNN_GRAPH_CONFIG_OPTION_PRIORITY:=3): 'QNN_GRAPH_CONFIG_OPTION_PRIORITY', (QNN_GRAPH_CONFIG_OPTION_PROFILE_HANDLE:=4): 'QNN_GRAPH_CONFIG_OPTION_PROFILE_HANDLE', (QNN_GRAPH_CONFIG_OPTION_SET_PROFILING_STATE:=5): 'QNN_GRAPH_CONFIG_OPTION_SET_PROFILING_STATE', (QNN_GRAPH_CONFIG_OPTION_SET_PROFILING_NUM_EXECUTIONS:=6): 'QNN_GRAPH_CONFIG_OPTION_SET_PROFILING_NUM_EXECUTIONS', (QNN_GRAPH_CONFIG_OPTION_ENABLE_BINARY_SECTION_WEIGHTS_UPDATES:=7): 'QNN_GRAPH_CONFIG_OPTION_ENABLE_BINARY_SECTION_WEIGHTS_UPDATES', (QNN_GRAPH_CONFIG_OPTION_UNDEFINED:=2147483647): 'QNN_GRAPH_CONFIG_OPTION_UNDEFINED'}
QnnGraph_CustomConfig_t: TypeAlias = ctypes.c_void_p
Qnn_ProfileHandle_t: TypeAlias = ctypes.c_void_p
QnnGraph_ProfilingState_t: dict[int, str] = {(QNN_GRAPH_PROFILING_STATE_ENABLED:=1): 'QNN_GRAPH_PROFILING_STATE_ENABLED', (QNN_GRAPH_PROFILING_STATE_DISABLED:=2): 'QNN_GRAPH_PROFILING_STATE_DISABLED', (QNN_GRAPH_PROFILING_STATE_UNDEFINED:=2147483647): 'QNN_GRAPH_PROFILING_STATE_UNDEFINED'}
QnnGraph_Config_t.register_fields([('option', ctypes.c_uint32, 0), ('customConfig', QnnGraph_CustomConfig_t, 8), ('priority', ctypes.c_uint32, 8), ('profileHandle', Qnn_ProfileHandle_t, 8), ('profilingState', ctypes.c_uint32, 8), ('numProfilingExecutions', uint32_t, 8), ('enableBinarySectionWeightsUpdates', uint8_t, 8)])
QnnGraph_CreateFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[ctypes.c_char], c.POINTER[c.POINTER[QnnGraph_Config_t]], c.POINTER[ctypes.c_void_p]]]
QnnGraph_CreateSubgraphFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[ctypes.c_char], c.POINTER[ctypes.c_void_p]]]
QnnGraph_SetConfigFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnGraph_Config_t]]]]
QnnGraph_AddNodeFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, Qnn_OpConfig_t]]
QnnGraph_FinalizeFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]]
QnnGraph_RetrieveFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[ctypes.c_char], c.POINTER[ctypes.c_void_p]]]
QnnGraph_ExecuteFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[Qnn_Tensor_t], ctypes.c_uint32, c.POINTER[Qnn_Tensor_t], ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]]
@c.record
class Qnn_NotifyStatus_t(c.Struct):
  SIZE = 8
  error: int
Qnn_ErrorHandle_t: TypeAlias = ctypes.c_uint64
Qnn_NotifyStatus_t.register_fields([('error', Qnn_ErrorHandle_t, 0)])
QnnGraph_ExecuteAsyncFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[Qnn_Tensor_t], ctypes.c_uint32, c.POINTER[Qnn_Tensor_t], ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, c.CFUNCTYPE[None, [ctypes.c_void_p, Qnn_NotifyStatus_t]], ctypes.c_void_p]]
QnnTensor_CreateContextTensorFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[Qnn_Tensor_t]]]
QnnTensor_CreateGraphTensorFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[Qnn_Tensor_t]]]
QnnLog_Level_t: dict[int, str] = {(QNN_LOG_LEVEL_ERROR:=1): 'QNN_LOG_LEVEL_ERROR', (QNN_LOG_LEVEL_WARN:=2): 'QNN_LOG_LEVEL_WARN', (QNN_LOG_LEVEL_INFO:=3): 'QNN_LOG_LEVEL_INFO', (QNN_LOG_LEVEL_VERBOSE:=4): 'QNN_LOG_LEVEL_VERBOSE', (QNN_LOG_LEVEL_DEBUG:=5): 'QNN_LOG_LEVEL_DEBUG', (QNN_LOG_LEVEL_MAX:=2147483647): 'QNN_LOG_LEVEL_MAX'}
@c.record
class struct___va_list_tag(c.Struct):
  SIZE = 24
  gp_offset: int
  fp_offset: int
  overflow_arg_area: ctypes.c_void_p
  reg_save_area: ctypes.c_void_p
struct___va_list_tag.register_fields([('gp_offset', ctypes.c_uint32, 0), ('fp_offset', ctypes.c_uint32, 4), ('overflow_arg_area', ctypes.c_void_p, 8), ('reg_save_area', ctypes.c_void_p, 16)])
QnnLog_CreateFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [c.CFUNCTYPE[None, [c.POINTER[ctypes.c_char], ctypes.c_uint32, ctypes.c_uint64, c.POINTER[struct___va_list_tag]]], ctypes.c_uint32, c.POINTER[ctypes.c_void_p]]]
QnnLog_SetLogLevelFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_uint32]]
QnnLog_FreeFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p]]
QnnProfile_CreateFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_uint32, c.POINTER[ctypes.c_void_p]]]
@c.record
class QnnProfile_Config_t(c.Struct):
  SIZE = 16
  option: int
  customConfig: ctypes.c_void_p
  numMaxEvents: int
  enableOptrace: int
QnnProfile_ConfigOption_t: dict[int, str] = {(QNN_PROFILE_CONFIG_OPTION_CUSTOM:=0): 'QNN_PROFILE_CONFIG_OPTION_CUSTOM', (QNN_PROFILE_CONFIG_OPTION_MAX_EVENTS:=1): 'QNN_PROFILE_CONFIG_OPTION_MAX_EVENTS', (QNN_PROFILE_CONFIG_OPTION_ENABLE_OPTRACE:=2): 'QNN_PROFILE_CONFIG_OPTION_ENABLE_OPTRACE', (QNN_PROFILE_CONFIG_OPTION_UNDEFINED:=2147483647): 'QNN_PROFILE_CONFIG_OPTION_UNDEFINED'}
QnnProfile_CustomConfig_t: TypeAlias = ctypes.c_void_p
QnnProfile_Config_t.register_fields([('option', ctypes.c_uint32, 0), ('customConfig', QnnProfile_CustomConfig_t, 8), ('numMaxEvents', uint64_t, 8), ('enableOptrace', uint8_t, 8)])
QnnProfile_SetConfigFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnProfile_Config_t]]]]
QnnProfile_GetEventsFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[ctypes.c_uint64]], c.POINTER[ctypes.c_uint32]]]
QnnProfile_GetSubEventsFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_uint64, c.POINTER[c.POINTER[ctypes.c_uint64]], c.POINTER[ctypes.c_uint32]]]
@c.record
class QnnProfile_EventData_t(c.Struct):
  SIZE = 24
  type: int
  unit: int
  value: int
  identifier: c.POINTER[ctypes.c_char]
QnnProfile_EventType_t: TypeAlias = ctypes.c_uint32
QnnProfile_EventUnit_t: TypeAlias = ctypes.c_uint32
QnnProfile_EventValue_t: TypeAlias = ctypes.c_uint64
QnnProfile_EventData_t.register_fields([('type', QnnProfile_EventType_t, 0), ('unit', QnnProfile_EventUnit_t, 4), ('value', QnnProfile_EventValue_t, 8), ('identifier', c.POINTER[ctypes.c_char], 16)])
QnnProfile_GetEventDataFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_uint64, c.POINTER[QnnProfile_EventData_t]]]
@c.record
class QnnProfile_ExtendedEventData_t(c.Struct):
  SIZE = 56
  version: int
  v1: QnnProfile_ExtendedEventDataV1_t
QnnProfile_ExtendedEventDataVersion_t: dict[int, str] = {(QNN_PROFILE_DATA_VERSION_1:=1): 'QNN_PROFILE_DATA_VERSION_1', (QNN_PROFILE_DATA_VERSION_UNDEFINED:=2147483647): 'QNN_PROFILE_DATA_VERSION_UNDEFINED'}
@c.record
class QnnProfile_ExtendedEventDataV1_t(c.Struct):
  SIZE = 48
  type: int
  unit: int
  value: Qnn_Scalar_t
  backendOpaqueObject: QnnProfile_BackendOpaqueObject_t
  timestamp: int
  identifier: c.POINTER[ctypes.c_char]
@c.record
class QnnProfile_BackendOpaqueObject_t(c.Struct):
  SIZE = 24
  opaqueObject: Qnn_OpaqueObject_t
  fileName: c.POINTER[ctypes.c_char]
@c.record
class Qnn_OpaqueObject_t(c.Struct):
  SIZE = 16
  data: ctypes.c_void_p
  len: int
Qnn_OpaqueObject_t.register_fields([('data', ctypes.c_void_p, 0), ('len', uint64_t, 8)])
QnnProfile_BackendOpaqueObject_t.register_fields([('opaqueObject', Qnn_OpaqueObject_t, 0), ('fileName', c.POINTER[ctypes.c_char], 16)])
QnnProfile_ExtendedEventDataV1_t.register_fields([('type', QnnProfile_EventType_t, 0), ('unit', QnnProfile_EventUnit_t, 4), ('value', Qnn_Scalar_t, 8), ('backendOpaqueObject', QnnProfile_BackendOpaqueObject_t, 8), ('timestamp', uint64_t, 32), ('identifier', c.POINTER[ctypes.c_char], 40)])
QnnProfile_ExtendedEventData_t.register_fields([('version', ctypes.c_uint32, 0), ('v1', QnnProfile_ExtendedEventDataV1_t, 8)])
QnnProfile_GetExtendedEventDataFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_uint64, c.POINTER[QnnProfile_ExtendedEventData_t]]]
QnnProfile_FreeFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p]]
@c.record
class Qnn_MemDescriptor_t(c.Struct):
  SIZE = 48
  memShape: Qnn_MemShape_t
  dataType: int
  memType: int
  ionInfo: Qnn_MemIonInfo_t
  customInfo: ctypes.c_void_p
  dmaBufInfo: Qnn_MemDmaBufInfo_t
@c.record
class Qnn_MemShape_t(c.Struct):
  SIZE = 24
  numDim: int
  dimSize: c.POINTER[ctypes.c_uint32]
  shapeConfig: c.POINTER[ctypes.c_char]
Qnn_MemShape_t.register_fields([('numDim', uint32_t, 0), ('dimSize', c.POINTER[uint32_t], 8), ('shapeConfig', c.POINTER[ctypes.c_char], 16)])
Qnn_MemType_t: dict[int, str] = {(QNN_MEM_TYPE_ION:=1): 'QNN_MEM_TYPE_ION', (QNN_MEM_TYPE_CUSTOM:=2): 'QNN_MEM_TYPE_CUSTOM', (QNN_MEM_TYPE_DMA_BUF:=3): 'QNN_MEM_TYPE_DMA_BUF', (QNN_MEM_TYPE_UNDEFINED:=2147483647): 'QNN_MEM_TYPE_UNDEFINED'}
@c.record
class Qnn_MemIonInfo_t(c.Struct):
  SIZE = 4
  fd: int
Qnn_MemIonInfo_t.register_fields([('fd', int32_t, 0)])
Qnn_MemInfoCustom_t: TypeAlias = ctypes.c_void_p
@c.record
class Qnn_MemDmaBufInfo_t(c.Struct):
  SIZE = 16
  fd: int
  data: ctypes.c_void_p
Qnn_MemDmaBufInfo_t.register_fields([('fd', int32_t, 0), ('data', ctypes.c_void_p, 8)])
Qnn_MemDescriptor_t.register_fields([('memShape', Qnn_MemShape_t, 0), ('dataType', ctypes.c_uint32, 24), ('memType', ctypes.c_uint32, 28), ('ionInfo', Qnn_MemIonInfo_t, 32), ('customInfo', Qnn_MemInfoCustom_t, 32), ('dmaBufInfo', Qnn_MemDmaBufInfo_t, 32)])
QnnMem_RegisterFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[Qnn_MemDescriptor_t], ctypes.c_uint32, c.POINTER[ctypes.c_void_p]]]
QnnMem_DeRegisterFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[ctypes.c_void_p], ctypes.c_uint32]]
@c.record
class QnnDevice_PlatformInfo_t(c.Struct):
  SIZE = 24
  version: int
  v1: QnnDevice_PlatformInfoV1_t
QnnDevice_PlatformInfoVersion_t: dict[int, str] = {(QNN_DEVICE_PLATFORM_INFO_VERSION_1:=1): 'QNN_DEVICE_PLATFORM_INFO_VERSION_1', (QNN_DEVICE_PLATFORM_INFO_VERSION_UNDEFINED:=2147483647): 'QNN_DEVICE_PLATFORM_INFO_VERSION_UNDEFINED'}
@c.record
class QnnDevice_PlatformInfoV1_t(c.Struct):
  SIZE = 16
  numHwDevices: int
  hwDevices: c.POINTER[QnnDevice_HardwareDeviceInfo_t]
@c.record
class QnnDevice_HardwareDeviceInfo_t(c.Struct):
  SIZE = 40
  version: int
  v1: QnnDevice_HardwareDeviceInfoV1_t
QnnDevice_HardwareDeviceInfoVersion_t: dict[int, str] = {(QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1:=1): 'QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1', (QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_UNDEFINED:=2147483647): 'QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_UNDEFINED'}
@c.record
class QnnDevice_HardwareDeviceInfoV1_t(c.Struct):
  SIZE = 32
  deviceId: int
  deviceType: int
  numCores: int
  cores: c.POINTER[QnnDevice_CoreInfo_t]
  deviceInfoExtension: c.POINTER[struct__QnnDevice_DeviceInfoExtension_t]
@c.record
class QnnDevice_CoreInfo_t(c.Struct):
  SIZE = 24
  version: int
  v1: QnnDevice_CoreInfoV1_t
QnnDevice_CoreInfoVersion_t: dict[int, str] = {(QNN_DEVICE_CORE_INFO_VERSION_1:=1): 'QNN_DEVICE_CORE_INFO_VERSION_1', (QNN_DEVICE_CORE_INFO_VERSION_UNDEFINED:=2147483647): 'QNN_DEVICE_CORE_INFO_VERSION_UNDEFINED'}
@c.record
class QnnDevice_CoreInfoV1_t(c.Struct):
  SIZE = 16
  coreId: int
  coreType: int
  coreInfoExtension: c.POINTER[struct__QnnDevice_CoreInfoExtension_t]
class struct__QnnDevice_CoreInfoExtension_t(c.Struct): pass
QnnDevice_CoreInfoExtension_t: TypeAlias = c.POINTER[struct__QnnDevice_CoreInfoExtension_t]
QnnDevice_CoreInfoV1_t.register_fields([('coreId', uint32_t, 0), ('coreType', uint32_t, 4), ('coreInfoExtension', QnnDevice_CoreInfoExtension_t, 8)])
QnnDevice_CoreInfo_t.register_fields([('version', ctypes.c_uint32, 0), ('v1', QnnDevice_CoreInfoV1_t, 8)])
class struct__QnnDevice_DeviceInfoExtension_t(c.Struct): pass
QnnDevice_DeviceInfoExtension_t: TypeAlias = c.POINTER[struct__QnnDevice_DeviceInfoExtension_t]
QnnDevice_HardwareDeviceInfoV1_t.register_fields([('deviceId', uint32_t, 0), ('deviceType', uint32_t, 4), ('numCores', uint32_t, 8), ('cores', c.POINTER[QnnDevice_CoreInfo_t], 16), ('deviceInfoExtension', QnnDevice_DeviceInfoExtension_t, 24)])
QnnDevice_HardwareDeviceInfo_t.register_fields([('version', ctypes.c_uint32, 0), ('v1', QnnDevice_HardwareDeviceInfoV1_t, 8)])
QnnDevice_PlatformInfoV1_t.register_fields([('numHwDevices', uint32_t, 0), ('hwDevices', c.POINTER[QnnDevice_HardwareDeviceInfo_t], 8)])
QnnDevice_PlatformInfo_t.register_fields([('version', ctypes.c_uint32, 0), ('v1', QnnDevice_PlatformInfoV1_t, 8)])
QnnDevice_GetPlatformInfoFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnDevice_PlatformInfo_t]]]]
QnnDevice_FreePlatformInfoFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[QnnDevice_PlatformInfo_t]]]
@c.record
class struct__QnnDevice_Infrastructure_t(c.Struct):
  SIZE = 40
  infraType: int
  perfInfra: QnnHtpDevice_PerfInfrastructure_t
QnnDevice_GetInfrastructureFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[c.POINTER[struct__QnnDevice_Infrastructure_t]]]]
@c.record
class QnnDevice_Config_t(c.Struct):
  SIZE = 16
  option: int
  customConfig: ctypes.c_void_p
  hardwareInfo: c.POINTER[QnnDevice_PlatformInfo_t]
QnnDevice_ConfigOption_t: dict[int, str] = {(QNN_DEVICE_CONFIG_OPTION_CUSTOM:=0): 'QNN_DEVICE_CONFIG_OPTION_CUSTOM', (QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO:=1): 'QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO', (QNN_DEVICE_CONFIG_OPTION_UNDEFINED:=2147483647): 'QNN_DEVICE_CONFIG_OPTION_UNDEFINED'}
QnnDevice_CustomConfig_t: TypeAlias = ctypes.c_void_p
QnnDevice_Config_t.register_fields([('option', ctypes.c_uint32, 0), ('customConfig', QnnDevice_CustomConfig_t, 8), ('hardwareInfo', c.POINTER[QnnDevice_PlatformInfo_t], 8)])
QnnDevice_CreateFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnDevice_Config_t]], c.POINTER[ctypes.c_void_p]]]
QnnDevice_SetConfigFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnDevice_Config_t]]]]
QnnDevice_GetInfoFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnDevice_PlatformInfo_t]]]]
QnnDevice_FreeFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p]]
@c.record
class QnnSignal_Config_t(c.Struct):
  SIZE = 16
  option: int
  customConfig: ctypes.c_void_p
  timeoutDurationUs: int
QnnSignal_ConfigOption_t: dict[int, str] = {(QNN_SIGNAL_CONFIG_OPTION_CUSTOM:=0): 'QNN_SIGNAL_CONFIG_OPTION_CUSTOM', (QNN_SIGNAL_CONFIG_OPTION_ABORT:=1): 'QNN_SIGNAL_CONFIG_OPTION_ABORT', (QNN_SIGNAL_CONFIG_OPTION_TIMEOUT:=2): 'QNN_SIGNAL_CONFIG_OPTION_TIMEOUT', (QNN_SIGNAL_CONFIG_UNDEFINED:=2147483647): 'QNN_SIGNAL_CONFIG_UNDEFINED'}
QnnSignal_CustomConfig_t: TypeAlias = ctypes.c_void_p
QnnSignal_Config_t.register_fields([('option', ctypes.c_uint32, 0), ('customConfig', QnnSignal_CustomConfig_t, 8), ('timeoutDurationUs', uint64_t, 8)])
QnnSignal_CreateFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnSignal_Config_t]], c.POINTER[ctypes.c_void_p]]]
QnnSignal_SetConfigFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnSignal_Config_t]]]]
QnnSignal_TriggerFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p]]
QnnSignal_FreeFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p]]
QnnError_GetMessageFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_uint64, c.POINTER[c.POINTER[ctypes.c_char]]]]
QnnError_GetVerboseMessageFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_uint64, c.POINTER[c.POINTER[ctypes.c_char]]]]
QnnError_FreeVerboseMessageFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[ctypes.c_char]]]
@c.record
class QnnGraph_ExecuteEnvironment_t(c.Struct):
  SIZE = 48
  option: int
  tensorSet: Qnn_TensorSet_t
QnnGraph_ExecuteEnvironmentOption_t: dict[int, str] = {(QNN_GRAPH_EXECUTE_ENVIRONMENT_OPTION_BIND_MEM_HANDLES:=0): 'QNN_GRAPH_EXECUTE_ENVIRONMENT_OPTION_BIND_MEM_HANDLES', (QNN_GRAPH_EXECUTE_ENVIRONMENT_OPTION_POPULATE_CLIENT_BUFS:=1): 'QNN_GRAPH_EXECUTE_ENVIRONMENT_OPTION_POPULATE_CLIENT_BUFS', (QNN_GRAPH_EXECUTE_ENVIRONMENT_OPTION_UNDEFINED:=2147483647): 'QNN_GRAPH_EXECUTE_ENVIRONMENT_OPTION_UNDEFINED'}
@c.record
class Qnn_TensorSet_t(c.Struct):
  SIZE = 40
  version: int
  v1: Qnn_TensorSetV1_t
Qnn_TensorSetVersion_t: dict[int, str] = {(QNN_TENSOR_SET_VERSION_1:=1): 'QNN_TENSOR_SET_VERSION_1', (QNN_TENSOR_SET_VERSION_UNDEFINED:=2147483647): 'QNN_TENSOR_SET_VERSION_UNDEFINED'}
@c.record
class Qnn_TensorSetV1_t(c.Struct):
  SIZE = 32
  numInputs: int
  inputs: c.POINTER[Qnn_Tensor_t]
  numOutputs: int
  outputs: c.POINTER[Qnn_Tensor_t]
Qnn_TensorSetV1_t.register_fields([('numInputs', uint32_t, 0), ('inputs', c.POINTER[Qnn_Tensor_t], 8), ('numOutputs', uint32_t, 16), ('outputs', c.POINTER[Qnn_Tensor_t], 24)])
Qnn_TensorSet_t.register_fields([('version', ctypes.c_uint32, 0), ('v1', Qnn_TensorSetV1_t, 8)])
QnnGraph_ExecuteEnvironment_t.register_fields([('option', ctypes.c_uint32, 0), ('tensorSet', Qnn_TensorSet_t, 8)])
QnnGraph_PrepareExecutionEnvironmentFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnGraph_ExecuteEnvironment_t]], ctypes.c_uint32]]
QnnGraph_ReleaseExecutionEnvironmentFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnGraph_ExecuteEnvironment_t]], ctypes.c_uint32]]
@c.record
class QnnGraph_Property_t(c.Struct):
  SIZE = 16
  option: int
  customProperty: ctypes.c_void_p
QnnGraph_PropertyOption_t: dict[int, str] = {(QNN_GRAPH_PROPERTY_OPTION_CUSTOM:=0): 'QNN_GRAPH_PROPERTY_OPTION_CUSTOM', (QNN_GRAPH_PROPERTY_OPTION_UNDEFINED:=2147483647): 'QNN_GRAPH_PROPERTY_OPTION_UNDEFINED'}
QnnGraph_CustomProperty_t: TypeAlias = ctypes.c_void_p
QnnGraph_Property_t.register_fields([('option', ctypes.c_uint32, 0), ('customProperty', QnnGraph_CustomProperty_t, 8)])
QnnGraph_GetPropertyFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnGraph_Property_t]]]]
QnnContext_ValidateBinaryFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, c.POINTER[c.POINTER[QnnContext_Config_t]], ctypes.c_void_p, ctypes.c_uint64]]
QnnContext_CreateFromBinaryWithSignalFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, c.POINTER[c.POINTER[QnnContext_Config_t]], ctypes.c_void_p, ctypes.c_uint64, c.POINTER[ctypes.c_void_p], ctypes.c_void_p, ctypes.c_void_p]]
@c.record
class QnnContext_Params_t(c.Struct):
  SIZE = 56
  version: int
  v1: QnnContext_ParamsV1_t
  v2: c.POINTER[QnnContext_ParamsV2_t]
QnnContext_ParamsVersion_t: dict[int, str] = {(QNN_CONTEXT_PARAMS_VERSION_1:=1): 'QNN_CONTEXT_PARAMS_VERSION_1', (QNN_CONTEXT_PARAMS_VERSION_2:=2): 'QNN_CONTEXT_PARAMS_VERSION_2', (QNN_CONTEXT_PARAMS_VERSION_UNDEFINED:=2147483647): 'QNN_CONTEXT_PARAMS_VERSION_UNDEFINED'}
@c.record
class QnnContext_ParamsV1_t(c.Struct):
  SIZE = 48
  config: c.POINTER[c.POINTER[QnnContext_Config_t]]
  binaryBuffer: ctypes.c_void_p
  binaryBufferSize: int
  profile: ctypes.c_void_p
  notifyFunc: c.CFUNCTYPE[None, [ctypes.c_void_p, ctypes.c_void_p, c.POINTER[ctypes.c_char], ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint64]]
  notifyParam: ctypes.c_void_p
Qnn_ContextBinarySize_t: TypeAlias = ctypes.c_uint64
QnnContext_createFromBinaryAsyncNotifyType_t: dict[int, str] = {(QNN_CONTEXT_NOTIFY_TYPE_GRAPH_INIT:=0): 'QNN_CONTEXT_NOTIFY_TYPE_GRAPH_INIT', (QNN_CONTEXT_NOTIFY_TYPE_CONTEXT_INIT:=1): 'QNN_CONTEXT_NOTIFY_TYPE_CONTEXT_INIT', (QNN_CONTEXT_NOTIFY_TYPE_UNDEFINED:=134217727): 'QNN_CONTEXT_NOTIFY_TYPE_UNDEFINED'}
QnnContext_createFromBinaryNotifyFn_t: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_void_p, ctypes.c_void_p, c.POINTER[ctypes.c_char], ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint64]]
QnnContext_ParamsV1_t.register_fields([('config', c.POINTER[c.POINTER[QnnContext_Config_t]], 0), ('binaryBuffer', ctypes.c_void_p, 8), ('binaryBufferSize', Qnn_ContextBinarySize_t, 16), ('profile', Qnn_ProfileHandle_t, 24), ('notifyFunc', QnnContext_createFromBinaryNotifyFn_t, 32), ('notifyParam', ctypes.c_void_p, 40)])
@c.record
class QnnContext_ParamsV2_t(c.Struct):
  SIZE = 56
  config: c.POINTER[c.POINTER[QnnContext_Config_t]]
  binaryBuffer: ctypes.c_void_p
  binaryBufferSize: int
  profile: ctypes.c_void_p
  notifyFunc: c.CFUNCTYPE[None, [ctypes.c_void_p, ctypes.c_void_p, c.POINTER[ctypes.c_char], ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint64]]
  notifyParam: ctypes.c_void_p
  callback: c.POINTER[Qnn_ContextBinaryCallback_t]
@c.record
class Qnn_ContextBinaryCallback_t(c.Struct):
  SIZE = 40
  type: int
  rawBufferCallback: Qnn_ContextBinaryRawBufferCallback_t
  dmaBufferCallback: Qnn_ContextBinaryDmaBufferCallback_t
Qnn_ContextBinaryCallbackType_t: dict[int, str] = {(QNN_CONTEXT_CALLBACK_RAW_BUFFER:=1): 'QNN_CONTEXT_CALLBACK_RAW_BUFFER', (QNN_CONTEXT_CALLBACK_DMA_BUFFER:=2): 'QNN_CONTEXT_CALLBACK_DMA_BUFFER', (QNN_CONTEXT_CALLBACK_UNDEFINED:=2147483647): 'QNN_CONTEXT_CALLBACK_UNDEFINED'}
@c.record
class Qnn_ContextBinaryRawBufferCallback_t(c.Struct):
  SIZE = 32
  version: int
  v1: Qnn_ContextBinaryRawBufferCallbackV1_t
Qnn_ContextBinaryRawBufferCallbackVersion_t: dict[int, str] = {(QNN_CONTEXT_CALLBACK_RAW_BUFFER_VERSION_1:=1): 'QNN_CONTEXT_CALLBACK_RAW_BUFFER_VERSION_1', (QNN_CONTEXT_CALLBACK_RAW_BUFFER_VERSION_UNDEFINED:=2147483647): 'QNN_CONTEXT_CALLBACK_RAW_BUFFER_VERSION_UNDEFINED'}
@c.record
class Qnn_ContextBinaryRawBufferCallbackV1_t(c.Struct):
  SIZE = 24
  dataProvide: c.CFUNCTYPE[ctypes.c_uint64, [Qnn_ContextBinaryDataRequest_t, c.POINTER[Qnn_ContextBinaryRawDataResponse_t], ctypes.c_void_p]]
  dataRelease: c.CFUNCTYPE[ctypes.c_uint64, [Qnn_ContextBinaryRawDataMem_t, ctypes.c_void_p]]
  notifyParam: ctypes.c_void_p
@c.record
class Qnn_ContextBinaryDataRequest_t(c.Struct):
  SIZE = 24
  offset: int
  size: int
  isBackendMappingNeeded: int
Qnn_ContextBinaryDataRequest_t.register_fields([('offset', Qnn_ContextBinarySize_t, 0), ('size', Qnn_ContextBinarySize_t, 8), ('isBackendMappingNeeded', uint8_t, 16)])
@c.record
class Qnn_ContextBinaryRawDataResponse_t(c.Struct):
  SIZE = 24
  memPtr: ctypes.c_void_p
  dataStartOffset: int
  alignedSize: int
Qnn_ContextBinaryRawDataResponse_t.register_fields([('memPtr', ctypes.c_void_p, 0), ('dataStartOffset', Qnn_ContextBinarySize_t, 8), ('alignedSize', Qnn_ContextBinarySize_t, 16)])
Qnn_ContextBinaryRawDataProviderFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [Qnn_ContextBinaryDataRequest_t, c.POINTER[Qnn_ContextBinaryRawDataResponse_t], ctypes.c_void_p]]
@c.record
class Qnn_ContextBinaryRawDataMem_t(c.Struct):
  SIZE = 16
  memPtr: ctypes.c_void_p
  memSize: int
Qnn_ContextBinaryRawDataMem_t.register_fields([('memPtr', ctypes.c_void_p, 0), ('memSize', Qnn_ContextBinarySize_t, 8)])
Qnn_ContextBinaryRawDataReleaseFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [Qnn_ContextBinaryRawDataMem_t, ctypes.c_void_p]]
Qnn_ContextBinaryRawBufferCallbackV1_t.register_fields([('dataProvide', Qnn_ContextBinaryRawDataProviderFn_t, 0), ('dataRelease', Qnn_ContextBinaryRawDataReleaseFn_t, 8), ('notifyParam', ctypes.c_void_p, 16)])
Qnn_ContextBinaryRawBufferCallback_t.register_fields([('version', ctypes.c_uint32, 0), ('v1', Qnn_ContextBinaryRawBufferCallbackV1_t, 8)])
@c.record
class Qnn_ContextBinaryDmaBufferCallback_t(c.Struct):
  SIZE = 32
  version: int
  v1: Qnn_ContextBinaryDmaBufferCallbackV1_t
Qnn_ContextBinaryDmaBufferCallbackVersion_t: dict[int, str] = {(QNN_CONTEXT_CALLBACK_DMA_BUFFER_VERSION_1:=1): 'QNN_CONTEXT_CALLBACK_DMA_BUFFER_VERSION_1', (QNN_CONTEXT_CALLBACK_DMA_BUFFER_VERSION_UNDEFINED:=2147483647): 'QNN_CONTEXT_CALLBACK_DMA_BUFFER_VERSION_UNDEFINED'}
@c.record
class Qnn_ContextBinaryDmaBufferCallbackV1_t(c.Struct):
  SIZE = 24
  dataProvide: c.CFUNCTYPE[ctypes.c_uint64, [Qnn_ContextBinaryDataRequest_t, c.POINTER[Qnn_ContextBinaryDmaDataResponse_t], ctypes.c_void_p]]
  dataRelease: c.CFUNCTYPE[ctypes.c_uint64, [Qnn_ContextBinaryDmaDataMem_t, ctypes.c_void_p]]
  notifyParam: ctypes.c_void_p
@c.record
class Qnn_ContextBinaryDmaDataResponse_t(c.Struct):
  SIZE = 32
  dmaBuffer: Qnn_MemDmaBufInfo_t
  dataStartOffset: int
  alignedSize: int
Qnn_ContextBinaryDmaDataResponse_t.register_fields([('dmaBuffer', Qnn_MemDmaBufInfo_t, 0), ('dataStartOffset', Qnn_ContextBinarySize_t, 16), ('alignedSize', Qnn_ContextBinarySize_t, 24)])
Qnn_ContextBinaryDmaDataProviderFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [Qnn_ContextBinaryDataRequest_t, c.POINTER[Qnn_ContextBinaryDmaDataResponse_t], ctypes.c_void_p]]
@c.record
class Qnn_ContextBinaryDmaDataMem_t(c.Struct):
  SIZE = 24
  dmaBuffer: Qnn_MemDmaBufInfo_t
  memSize: int
Qnn_ContextBinaryDmaDataMem_t.register_fields([('dmaBuffer', Qnn_MemDmaBufInfo_t, 0), ('memSize', Qnn_ContextBinarySize_t, 16)])
Qnn_ContextBinaryDmaDataReleaseFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [Qnn_ContextBinaryDmaDataMem_t, ctypes.c_void_p]]
Qnn_ContextBinaryDmaBufferCallbackV1_t.register_fields([('dataProvide', Qnn_ContextBinaryDmaDataProviderFn_t, 0), ('dataRelease', Qnn_ContextBinaryDmaDataReleaseFn_t, 8), ('notifyParam', ctypes.c_void_p, 16)])
Qnn_ContextBinaryDmaBufferCallback_t.register_fields([('version', ctypes.c_uint32, 0), ('v1', Qnn_ContextBinaryDmaBufferCallbackV1_t, 8)])
Qnn_ContextBinaryCallback_t.register_fields([('type', ctypes.c_uint32, 0), ('rawBufferCallback', Qnn_ContextBinaryRawBufferCallback_t, 8), ('dmaBufferCallback', Qnn_ContextBinaryDmaBufferCallback_t, 8)])
QnnContext_ParamsV2_t.register_fields([('config', c.POINTER[c.POINTER[QnnContext_Config_t]], 0), ('binaryBuffer', ctypes.c_void_p, 8), ('binaryBufferSize', Qnn_ContextBinarySize_t, 16), ('profile', Qnn_ProfileHandle_t, 24), ('notifyFunc', QnnContext_createFromBinaryNotifyFn_t, 32), ('notifyParam', ctypes.c_void_p, 40), ('callback', c.POINTER[Qnn_ContextBinaryCallback_t], 48)])
QnnContext_Params_t.register_fields([('version', ctypes.c_uint32, 0), ('v1', QnnContext_ParamsV1_t, 8), ('v2', c.POINTER[QnnContext_ParamsV2_t], 8)])
QnnContext_CreateFromBinaryListAsyncFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, c.POINTER[c.POINTER[QnnContext_Params_t]], c.POINTER[c.POINTER[QnnContext_Config_t]], ctypes.c_void_p]]
QnnTensor_UpdateGraphTensorsFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[Qnn_Tensor_t]], ctypes.c_uint64]]
QnnTensor_UpdateContextTensorsFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[Qnn_Tensor_t]], ctypes.c_uint64]]
QnnContext_SectionType_t: dict[int, str] = {(QNN_CONTEXT_SECTION_UPDATABLE:=1): 'QNN_CONTEXT_SECTION_UPDATABLE', (QNN_CONTEXT_SECTION_UPDATABLE_WEIGHTS:=2): 'QNN_CONTEXT_SECTION_UPDATABLE_WEIGHTS', (QNN_CONTEXT_SECTION_UPDATABLE_QUANT_PARAMS:=3): 'QNN_CONTEXT_SECTION_UPDATABLE_QUANT_PARAMS', (QNN_CONTEXT_SECTION_UNDEFINED:=2147483647): 'QNN_CONTEXT_SECTION_UNDEFINED'}
QnnContext_GetBinarySectionSizeFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, c.POINTER[ctypes.c_uint64]]]
@c.record
class QnnContext_Buffer_t(c.Struct):
  SIZE = 32
  version: int
  v1: QnnContext_BufferV1_t
QnnContext_BufferVersion_t: dict[int, str] = {(QNN_CONTEXT_BUFFER_VERSION_1:=1): 'QNN_CONTEXT_BUFFER_VERSION_1', (QNN_CONTEXT_BUFFER_VERSION_UNDEFINED:=2147483647): 'QNN_CONTEXT_BUFFER_VERSION_UNDEFINED'}
@c.record
class QnnContext_BufferV1_t(c.Struct):
  SIZE = 24
  memType: int
  binaryBuf: Qnn_BinaryBuffer_t
  memHandle: ctypes.c_void_p
QnnContext_MemType_t: dict[int, str] = {(QNN_CONTEXTMEMTYPE_RAW:=0): 'QNN_CONTEXTMEMTYPE_RAW', (QNN_CONTEXTMEMTYPE_MEMHANDLE:=1): 'QNN_CONTEXTMEMTYPE_MEMHANDLE', (QNN_CONTEXTMEMTYPE_UNDEFINED:=2147483647): 'QNN_CONTEXTMEMTYPE_UNDEFINED'}
@c.record
class Qnn_BinaryBuffer_t(c.Struct):
  SIZE = 16
  data: ctypes.c_void_p
  dataSize: int
Qnn_BinaryBuffer_t.register_fields([('data', ctypes.c_void_p, 0), ('dataSize', Qnn_ContextBinarySize_t, 8)])
QnnContext_BufferV1_t.register_fields([('memType', ctypes.c_uint32, 0), ('binaryBuf', Qnn_BinaryBuffer_t, 8), ('memHandle', Qnn_MemHandle_t, 8)])
QnnContext_Buffer_t.register_fields([('version', ctypes.c_uint32, 0), ('v1', QnnContext_BufferV1_t, 8)])
QnnContext_GetBinarySectionFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, c.POINTER[QnnContext_Buffer_t], c.POINTER[ctypes.c_uint64], ctypes.c_void_p, ctypes.c_void_p]]
QnnContext_ApplyBinarySectionFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, c.POINTER[QnnContext_Buffer_t], ctypes.c_void_p, ctypes.c_void_p]]
@c.record
class QnnBackend_Property_t(c.Struct):
  SIZE = 16
  option: int
  customProperty: ctypes.c_void_p
QnnBackend_PropertyOption_t: dict[int, str] = {(QNN_BACKEND_PROPERTY_OPTION_CUSTOM:=0): 'QNN_BACKEND_PROPERTY_OPTION_CUSTOM', (QNN_BACKEND_PROPERTY_OPTION_UNDEFINED:=2147483647): 'QNN_BACKEND_PROPERTY_OPTION_UNDEFINED'}
QnnBackend_CustomProperty_t: TypeAlias = ctypes.c_void_p
QnnBackend_Property_t.register_fields([('option', ctypes.c_uint32, 0), ('customProperty', QnnBackend_CustomProperty_t, 8)])
QnnBackend_GetPropertyFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnBackend_Property_t]]]]
@c.record
class QnnContext_Property_t(c.Struct):
  SIZE = 16
  option: int
  customProperty: ctypes.c_void_p
QnnContext_PropertyOption_t: dict[int, str] = {(QNN_CONTEXT_PROPERTY_OPTION_CUSTOM:=0): 'QNN_CONTEXT_PROPERTY_OPTION_CUSTOM', (QNN_CONTEXT_PROPERTY_OPTION_UNDEFINED:=2147483647): 'QNN_CONTEXT_PROPERTY_OPTION_UNDEFINED'}
QnnContext_CustomProperty_t: TypeAlias = ctypes.c_void_p
QnnContext_Property_t.register_fields([('option', ctypes.c_uint32, 0), ('customProperty', QnnContext_CustomProperty_t, 8)])
QnnContext_GetPropertyFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnContext_Property_t]]]]
QnnContext_GetIncrementalBinaryFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[ctypes.c_void_p], c.POINTER[ctypes.c_uint64], c.POINTER[ctypes.c_uint64]]]
QnnContext_ReleaseIncrementalBinaryFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]]
QnnContext_FinalizeFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p]]
@c.record
class QnnGlobalConfig_t(c.Struct):
  SIZE = 8
  option: int
  socModel: int
QnnGlobalConfig_ConfigOption_t: dict[int, str] = {(QNN_GLOBAL_CONFIG_OPTION_SOC_MODEL:=0): 'QNN_GLOBAL_CONFIG_OPTION_SOC_MODEL', (QNN_GLOBAL_CONFIG_OPTION_UNDEFINED:=2147483647): 'QNN_GLOBAL_CONFIG_OPTION_UNDEFINED'}
Qnn_SocModel_t: dict[int, str] = {(QNN_SOC_MODEL_UNKNOWN:=0): 'QNN_SOC_MODEL_UNKNOWN', (QNN_SOC_MODEL_SDM845:=1): 'QNN_SOC_MODEL_SDM845', (QNN_SOC_MODEL_SDM835:=2): 'QNN_SOC_MODEL_SDM835', (QNN_SOC_MODEL_SDM821:=3): 'QNN_SOC_MODEL_SDM821', (QNN_SOC_MODEL_SDM820:=4): 'QNN_SOC_MODEL_SDM820', (QNN_SOC_MODEL_SDM801:=5): 'QNN_SOC_MODEL_SDM801', (QNN_SOC_MODEL_SDM670:=6): 'QNN_SOC_MODEL_SDM670', (QNN_SOC_MODEL_SDM660:=7): 'QNN_SOC_MODEL_SDM660', (QNN_SOC_MODEL_SDM652:=8): 'QNN_SOC_MODEL_SDM652', (QNN_SOC_MODEL_SDM636:=9): 'QNN_SOC_MODEL_SDM636', (QNN_SOC_MODEL_SDM630:=10): 'QNN_SOC_MODEL_SDM630', (QNN_SOC_MODEL_SDM625:=11): 'QNN_SOC_MODEL_SDM625', (QNN_SOC_MODEL_SDM855:=12): 'QNN_SOC_MODEL_SDM855', (QNN_SOC_MODEL_SDM710:=13): 'QNN_SOC_MODEL_SDM710', (QNN_SOC_MODEL_SDM632:=15): 'QNN_SOC_MODEL_SDM632', (QNN_SOC_MODEL_SM6150:=16): 'QNN_SOC_MODEL_SM6150', (QNN_SOC_MODEL_SM7150:=17): 'QNN_SOC_MODEL_SM7150', (QNN_SOC_MODEL_QCS405:=18): 'QNN_SOC_MODEL_QCS405', (QNN_SOC_MODEL_SM6125:=19): 'QNN_SOC_MODEL_SM6125', (QNN_SOC_MODEL_QCS403:=20): 'QNN_SOC_MODEL_QCS403', (QNN_SOC_MODEL_SDM865:=21): 'QNN_SOC_MODEL_SDM865', (QNN_SOC_MODEL_IPQ6018:=23): 'QNN_SOC_MODEL_IPQ6018', (QNN_SOC_MODEL_IPQ6028:=24): 'QNN_SOC_MODEL_IPQ6028', (QNN_SOC_MODEL_SM7250:=25): 'QNN_SOC_MODEL_SM7250', (QNN_SOC_MODEL_SA8195:=26): 'QNN_SOC_MODEL_SA8195', (QNN_SOC_MODEL_SM6250:=27): 'QNN_SOC_MODEL_SM6250', (QNN_SOC_MODEL_SM4250:=28): 'QNN_SOC_MODEL_SM4250', (QNN_SOC_MODEL_SM6350:=29): 'QNN_SOC_MODEL_SM6350', (QNN_SOC_MODEL_SM8350:=30): 'QNN_SOC_MODEL_SM8350', (QNN_SOC_MODEL_SM4350:=31): 'QNN_SOC_MODEL_SM4350', (QNN_SOC_MODEL_SM7350:=32): 'QNN_SOC_MODEL_SM7350', (QNN_SOC_MODEL_QCS410:=33): 'QNN_SOC_MODEL_QCS410', (QNN_SOC_MODEL_SM8325:=34): 'QNN_SOC_MODEL_SM8325', (QNN_SOC_MODEL_SM7325:=35): 'QNN_SOC_MODEL_SM7325', (QNN_SOC_MODEL_SM8450:=36): 'QNN_SOC_MODEL_SM8450', (QNN_SOC_MODEL_SC8280X:=37): 'QNN_SOC_MODEL_SC8280X', (QNN_SOC_MODEL_SM7315:=38): 'QNN_SOC_MODEL_SM7315', (QNN_SOC_MODEL_SA8295:=39): 'QNN_SOC_MODEL_SA8295', (QNN_SOC_MODEL_SM6225:=40): 'QNN_SOC_MODEL_SM6225', (QNN_SOC_MODEL_SM7450:=41): 'QNN_SOC_MODEL_SM7450', (QNN_SOC_MODEL_SM8475:=42): 'QNN_SOC_MODEL_SM8475', (QNN_SOC_MODEL_SM8550:=43): 'QNN_SOC_MODEL_SM8550', (QNN_SOC_MODEL_SXR1230P:=45): 'QNN_SOC_MODEL_SXR1230P', (QNN_SOC_MODEL_SSG2115P:=46): 'QNN_SOC_MODEL_SSG2115P', (QNN_SOC_MODEL_STP6225P:=47): 'QNN_SOC_MODEL_STP6225P', (QNN_SOC_MODEL_QCS6125:=48): 'QNN_SOC_MODEL_QCS6125', (QNN_SOC_MODEL_QRB4210:=49): 'QNN_SOC_MODEL_QRB4210', (QNN_SOC_MODEL_SM6450:=50): 'QNN_SOC_MODEL_SM6450', (QNN_SOC_MODEL_QCS7230:=51): 'QNN_SOC_MODEL_QCS7230', (QNN_SOC_MODEL_SA8255:=52): 'QNN_SOC_MODEL_SA8255', (QNN_SOC_MODEL_SXR2230P:=53): 'QNN_SOC_MODEL_SXR2230P', (QNN_SOC_MODEL_SM7475:=54): 'QNN_SOC_MODEL_SM7475', (QNN_SOC_MODEL_SM4375:=55): 'QNN_SOC_MODEL_SM4375', (QNN_SOC_MODEL_QCM4325:=56): 'QNN_SOC_MODEL_QCM4325', (QNN_SOC_MODEL_SM8650:=57): 'QNN_SOC_MODEL_SM8650', (QNN_SOC_MODEL_SSG2125P:=58): 'QNN_SOC_MODEL_SSG2125P', (QNN_SOC_MODEL_SM4450:=59): 'QNN_SOC_MODEL_SM4450', (QNN_SOC_MODEL_SC8380XP:=60): 'QNN_SOC_MODEL_SC8380XP', (QNN_SOC_MODEL_SM7435:=61): 'QNN_SOC_MODEL_SM7435', (QNN_SOC_MODEL_SA8540:=62): 'QNN_SOC_MODEL_SA8540', (QNN_SOC_MODEL_AIC100:=63): 'QNN_SOC_MODEL_AIC100', (QNN_SOC_MODEL_SM7550:=64): 'QNN_SOC_MODEL_SM7550', (QNN_SOC_MODEL_SM6450Q:=65): 'QNN_SOC_MODEL_SM6450Q', (QNN_SOC_MODEL_QCS8550:=66): 'QNN_SOC_MODEL_QCS8550', (QNN_SOC_MODEL_SA8620P:=67): 'QNN_SOC_MODEL_SA8620P', (QNN_SOC_MODEL_SM8635:=68): 'QNN_SOC_MODEL_SM8635', (QNN_SOC_MODEL_SM8750:=69): 'QNN_SOC_MODEL_SM8750', (QNN_SOC_MODEL_SM7675:=70): 'QNN_SOC_MODEL_SM7675', (QNN_SOC_MODEL_SM4635:=71): 'QNN_SOC_MODEL_SM4635', (QNN_SOC_MODEL_SA8797:=72): 'QNN_SOC_MODEL_SA8797', (QNN_SOC_MODEL_SM7635:=73): 'QNN_SOC_MODEL_SM7635', (QNN_SOC_MODEL_SM6650:=74): 'QNN_SOC_MODEL_SM6650', (QNN_SOC_MODEL_SXR2330P:=75): 'QNN_SOC_MODEL_SXR2330P', (QNN_SOC_MODEL_SM6475:=76): 'QNN_SOC_MODEL_SM6475', (QNN_SOC_MODEL_QCS9100:=77): 'QNN_SOC_MODEL_QCS9100', (QNN_SOC_MODEL_QCM6690:=78): 'QNN_SOC_MODEL_QCM6690', (QNN_SOC_MODEL_IPQ9574:=79): 'QNN_SOC_MODEL_IPQ9574', (QNN_SOC_MODEL_IPQ5404:=80): 'QNN_SOC_MODEL_IPQ5404', (QNN_SOC_MODEL_IPQ5424:=81): 'QNN_SOC_MODEL_IPQ5424', (QNN_SOC_MODEL_QCS8300:=82): 'QNN_SOC_MODEL_QCS8300', (QNN_SOC_MODEL_QCS2290:=83): 'QNN_SOC_MODEL_QCS2290', (QNN_SOC_MODEL_SA525M:=84): 'QNN_SOC_MODEL_SA525M', (QNN_SOC_MODEL_SM8735:=85): 'QNN_SOC_MODEL_SM8735', (QNN_SOC_MODEL_SM7750:=86): 'QNN_SOC_MODEL_SM7750', (QNN_SOC_MODEL_SM8850:=87): 'QNN_SOC_MODEL_SM8850', (QNN_SOC_MODEL_DYNAMIC_SDM:=2147483647): 'QNN_SOC_MODEL_DYNAMIC_SDM'}
QnnGlobalConfig_t.register_fields([('option', ctypes.c_uint32, 0), ('socModel', ctypes.c_uint32, 4)])
QnnGlobalConfig_SetFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[c.POINTER[QnnGlobalConfig_t]]]]
QnnContext_CreateFromBinaryWithCallbackFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, c.POINTER[c.POINTER[QnnContext_Config_t]], c.POINTER[Qnn_ContextBinaryCallback_t], ctypes.c_void_p, ctypes.c_uint64, c.POINTER[ctypes.c_void_p], ctypes.c_void_p, ctypes.c_void_p]]
QnnContext_GetBinarySectionUpdateFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[QnnContext_Buffer_t], c.POINTER[QnnContext_Buffer_t], c.POINTER[c.POINTER[Qnn_Tensor_t]], ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c.POINTER[QnnContext_Buffer_t]]]
QnnContext_FreeBinarySectionUpdateFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [QnnContext_Buffer_t, ctypes.c_void_p]]
QnnInterface_ImplementationV2_32_t.register_fields([('propertyHasCapability', QnnProperty_HasCapabilityFn_t, 0), ('backendCreate', QnnBackend_CreateFn_t, 8), ('backendSetConfig', QnnBackend_SetConfigFn_t, 16), ('backendGetApiVersion', QnnBackend_GetApiVersionFn_t, 24), ('backendGetBuildId', QnnBackend_GetBuildIdFn_t, 32), ('backendRegisterOpPackage', QnnBackend_RegisterOpPackageFn_t, 40), ('backendGetSupportedOperations', QnnBackend_GetSupportedOperationsFn_t, 48), ('backendValidateOpConfig', QnnBackend_ValidateOpConfigFn_t, 56), ('backendFree', QnnBackend_FreeFn_t, 64), ('contextCreate', QnnContext_CreateFn_t, 72), ('contextSetConfig', QnnContext_SetConfigFn_t, 80), ('contextGetBinarySize', QnnContext_GetBinarySizeFn_t, 88), ('contextGetBinary', QnnContext_GetBinaryFn_t, 96), ('contextCreateFromBinary', QnnContext_CreateFromBinaryFn_t, 104), ('contextFree', QnnContext_FreeFn_t, 112), ('graphCreate', QnnGraph_CreateFn_t, 120), ('graphCreateSubgraph', QnnGraph_CreateSubgraphFn_t, 128), ('graphSetConfig', QnnGraph_SetConfigFn_t, 136), ('graphAddNode', QnnGraph_AddNodeFn_t, 144), ('graphFinalize', QnnGraph_FinalizeFn_t, 152), ('graphRetrieve', QnnGraph_RetrieveFn_t, 160), ('graphExecute', QnnGraph_ExecuteFn_t, 168), ('graphExecuteAsync', QnnGraph_ExecuteAsyncFn_t, 176), ('tensorCreateContextTensor', QnnTensor_CreateContextTensorFn_t, 184), ('tensorCreateGraphTensor', QnnTensor_CreateGraphTensorFn_t, 192), ('logCreate', QnnLog_CreateFn_t, 200), ('logSetLogLevel', QnnLog_SetLogLevelFn_t, 208), ('logFree', QnnLog_FreeFn_t, 216), ('profileCreate', QnnProfile_CreateFn_t, 224), ('profileSetConfig', QnnProfile_SetConfigFn_t, 232), ('profileGetEvents', QnnProfile_GetEventsFn_t, 240), ('profileGetSubEvents', QnnProfile_GetSubEventsFn_t, 248), ('profileGetEventData', QnnProfile_GetEventDataFn_t, 256), ('profileGetExtendedEventData', QnnProfile_GetExtendedEventDataFn_t, 264), ('profileFree', QnnProfile_FreeFn_t, 272), ('memRegister', QnnMem_RegisterFn_t, 280), ('memDeRegister', QnnMem_DeRegisterFn_t, 288), ('deviceGetPlatformInfo', QnnDevice_GetPlatformInfoFn_t, 296), ('deviceFreePlatformInfo', QnnDevice_FreePlatformInfoFn_t, 304), ('deviceGetInfrastructure', QnnDevice_GetInfrastructureFn_t, 312), ('deviceCreate', QnnDevice_CreateFn_t, 320), ('deviceSetConfig', QnnDevice_SetConfigFn_t, 328), ('deviceGetInfo', QnnDevice_GetInfoFn_t, 336), ('deviceFree', QnnDevice_FreeFn_t, 344), ('signalCreate', QnnSignal_CreateFn_t, 352), ('signalSetConfig', QnnSignal_SetConfigFn_t, 360), ('signalTrigger', QnnSignal_TriggerFn_t, 368), ('signalFree', QnnSignal_FreeFn_t, 376), ('errorGetMessage', QnnError_GetMessageFn_t, 384), ('errorGetVerboseMessage', QnnError_GetVerboseMessageFn_t, 392), ('errorFreeVerboseMessage', QnnError_FreeVerboseMessageFn_t, 400), ('graphPrepareExecutionEnvironment', QnnGraph_PrepareExecutionEnvironmentFn_t, 408), ('graphReleaseExecutionEnvironment', QnnGraph_ReleaseExecutionEnvironmentFn_t, 416), ('graphGetProperty', QnnGraph_GetPropertyFn_t, 424), ('contextValidateBinary', QnnContext_ValidateBinaryFn_t, 432), ('contextCreateFromBinaryWithSignal', QnnContext_CreateFromBinaryWithSignalFn_t, 440), ('contextCreateFromBinaryListAsync', QnnContext_CreateFromBinaryListAsyncFn_t, 448), ('tensorUpdateGraphTensors', QnnTensor_UpdateGraphTensorsFn_t, 456), ('tensorUpdateContextTensors', QnnTensor_UpdateContextTensorsFn_t, 464), ('contextGetBinarySectionSize', QnnContext_GetBinarySectionSizeFn_t, 472), ('contextGetBinarySection', QnnContext_GetBinarySectionFn_t, 480), ('contextApplyBinarySection', QnnContext_ApplyBinarySectionFn_t, 488), ('backendGetProperty', QnnBackend_GetPropertyFn_t, 496), ('contextGetProperty', QnnContext_GetPropertyFn_t, 504), ('contextGetIncrementalBinary', QnnContext_GetIncrementalBinaryFn_t, 512), ('contextReleaseIncrementalBinary', QnnContext_ReleaseIncrementalBinaryFn_t, 520), ('contextFinalize', QnnContext_FinalizeFn_t, 528), ('globalConfigSet', QnnGlobalConfig_SetFn_t, 536), ('contextCreateFromBinaryWithCallback', QnnContext_CreateFromBinaryWithCallbackFn_t, 544), ('contextGetBinarySectionUpdate', QnnContext_GetBinarySectionUpdateFn_t, 552), ('contextFreeBinarySectionUpdate', QnnContext_FreeBinarySectionUpdateFn_t, 560)])
QnnInterface_t.register_fields([('backendId', uint32_t, 0), ('providerName', c.POINTER[ctypes.c_char], 8), ('apiVersion', Qnn_ApiVersion_t, 16), ('v2_32', QnnInterface_ImplementationV2_32_t, 40)])
@c.record
class QnnSystemInterface_t(c.Struct):
  SIZE = 160
  backendId: int
  providerName: c.POINTER[ctypes.c_char]
  systemApiVersion: Qnn_Version_t
  v1_7: QnnSystemInterface_ImplementationV1_7_t
@c.record
class QnnSystemInterface_ImplementationV1_7_t(c.Struct):
  SIZE = 128
  systemContextCreate: c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[ctypes.c_void_p]]]
  systemContextGetBinaryInfo: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, c.POINTER[c.POINTER[QnnSystemContext_BinaryInfo_t]], c.POINTER[ctypes.c_uint64]]]
  systemContextGetMetaData: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, c.POINTER[c.POINTER[QnnSystemContext_BinaryInfo_t]]]]
  systemContextFree: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p]]
  systemTensorGetMemoryFootprint: c.CFUNCTYPE[ctypes.c_uint64, [Qnn_Tensor_t, c.POINTER[ctypes.c_uint64]]]
  systemLogCreate: c.CFUNCTYPE[ctypes.c_uint64, [c.CFUNCTYPE[None, [c.POINTER[ctypes.c_char], ctypes.c_uint32, ctypes.c_uint64, c.POINTER[struct___va_list_tag]]], ctypes.c_uint32, c.POINTER[ctypes.c_void_p]]]
  systemLogSetLogLevel: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_uint32]]
  systemLogFree: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p]]
  systemDlcCreateFromFile: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[ctypes.c_char], c.POINTER[ctypes.c_void_p]]]
  systemDlcCreateFromBinary: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[ctypes.c_ubyte], ctypes.c_uint64, c.POINTER[ctypes.c_void_p]]]
  systemDlcComposeGraphs: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnSystemDlc_GraphConfigInfo_t]], ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, QnnInterface_t, ctypes.c_uint32, c.POINTER[c.POINTER[QnnSystemContext_GraphInfo_t]], c.POINTER[ctypes.c_uint32]]]
  systemDlcGetOpMappings: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[Qnn_OpMapping_t]], c.POINTER[ctypes.c_uint32]]]
  systemDlcFree: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p]]
  systemProfileCreateSerializationTarget: c.CFUNCTYPE[ctypes.c_uint64, [QnnSystemProfile_SerializationTarget_t, c.POINTER[QnnSystemProfile_SerializationTargetConfig_t], ctypes.c_uint32, c.POINTER[ctypes.c_void_p]]]
  systemProfileSerializeEventData: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnSystemProfile_ProfileData_t]], ctypes.c_uint32]]
  systemProfileFreeSerializationTarget: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p]]
QnnSystemContext_CreateFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[ctypes.c_void_p]]]
@c.record
class QnnSystemContext_BinaryInfo_t(c.Struct):
  SIZE = 144
  version: int
  contextBinaryInfoV1: QnnSystemContext_BinaryInfoV1_t
  contextBinaryInfoV2: QnnSystemContext_BinaryInfoV2_t
  contextBinaryInfoV3: QnnSystemContext_BinaryInfoV3_t
QnnSystemContext_BinaryInfoVersion_t: dict[int, str] = {(QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:=1): 'QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1', (QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:=2): 'QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2', (QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:=3): 'QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3', (QNN_SYSTEM_CONTEXT_BINARY_INFO_UNDEFINED:=2147483647): 'QNN_SYSTEM_CONTEXT_BINARY_INFO_UNDEFINED'}
@c.record
class QnnSystemContext_BinaryInfoV1_t(c.Struct):
  SIZE = 128
  backendId: int
  buildId: c.POINTER[ctypes.c_char]
  coreApiVersion: Qnn_Version_t
  backendApiVersion: Qnn_Version_t
  socVersion: c.POINTER[ctypes.c_char]
  hwInfoBlobVersion: Qnn_Version_t
  contextBlobVersion: Qnn_Version_t
  hwInfoBlobSize: int
  hwInfoBlob: ctypes.c_void_p
  contextBlobSize: int
  numContextTensors: int
  contextTensors: c.POINTER[Qnn_Tensor_t]
  numGraphs: int
  graphs: c.POINTER[QnnSystemContext_GraphInfo_t]
@c.record
class QnnSystemContext_GraphInfo_t(c.Struct):
  SIZE = 88
  version: int
  graphInfoV1: QnnSystemContext_GraphInfoV1_t
  graphInfoV2: QnnSystemContext_GraphInfoV2_t
  graphInfoV3: QnnSystemContext_GraphInfoV3_t
QnnSystemContext_GraphInfoVersion_t: dict[int, str] = {(QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1:=1): 'QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1', (QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2:=2): 'QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2', (QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3:=3): 'QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3', (QNN_SYSTEM_CONTEXT_GRAPH_INFO_UNDEFINED:=2147483647): 'QNN_SYSTEM_CONTEXT_GRAPH_INFO_UNDEFINED'}
@c.record
class QnnSystemContext_GraphInfoV1_t(c.Struct):
  SIZE = 40
  graphName: c.POINTER[ctypes.c_char]
  numGraphInputs: int
  graphInputs: c.POINTER[Qnn_Tensor_t]
  numGraphOutputs: int
  graphOutputs: c.POINTER[Qnn_Tensor_t]
QnnSystemContext_GraphInfoV1_t.register_fields([('graphName', c.POINTER[ctypes.c_char], 0), ('numGraphInputs', uint32_t, 8), ('graphInputs', c.POINTER[Qnn_Tensor_t], 16), ('numGraphOutputs', uint32_t, 24), ('graphOutputs', c.POINTER[Qnn_Tensor_t], 32)])
@c.record
class QnnSystemContext_GraphInfoV2_t(c.Struct):
  SIZE = 56
  graphName: c.POINTER[ctypes.c_char]
  numGraphInputs: int
  graphInputs: c.POINTER[Qnn_Tensor_t]
  numGraphOutputs: int
  graphOutputs: c.POINTER[Qnn_Tensor_t]
  numUpdateableTensors: int
  updateableTensors: c.POINTER[Qnn_Tensor_t]
QnnSystemContext_GraphInfoV2_t.register_fields([('graphName', c.POINTER[ctypes.c_char], 0), ('numGraphInputs', uint32_t, 8), ('graphInputs', c.POINTER[Qnn_Tensor_t], 16), ('numGraphOutputs', uint32_t, 24), ('graphOutputs', c.POINTER[Qnn_Tensor_t], 32), ('numUpdateableTensors', uint32_t, 40), ('updateableTensors', c.POINTER[Qnn_Tensor_t], 48)])
@c.record
class QnnSystemContext_GraphInfoV3_t(c.Struct):
  SIZE = 80
  graphName: c.POINTER[ctypes.c_char]
  numGraphInputs: int
  graphInputs: c.POINTER[Qnn_Tensor_t]
  numGraphOutputs: int
  graphOutputs: c.POINTER[Qnn_Tensor_t]
  numUpdateableTensors: int
  updateableTensors: c.POINTER[Qnn_Tensor_t]
  graphBlobInfoSize: int
  graphBlobInfo: ctypes.c_void_p
  startOpIndex: int
  endOpIndex: int
QnnSystemContext_GraphInfoV3_t.register_fields([('graphName', c.POINTER[ctypes.c_char], 0), ('numGraphInputs', uint32_t, 8), ('graphInputs', c.POINTER[Qnn_Tensor_t], 16), ('numGraphOutputs', uint32_t, 24), ('graphOutputs', c.POINTER[Qnn_Tensor_t], 32), ('numUpdateableTensors', uint32_t, 40), ('updateableTensors', c.POINTER[Qnn_Tensor_t], 48), ('graphBlobInfoSize', uint32_t, 56), ('graphBlobInfo', ctypes.c_void_p, 64), ('startOpIndex', uint32_t, 72), ('endOpIndex', uint32_t, 76)])
QnnSystemContext_GraphInfo_t.register_fields([('version', ctypes.c_uint32, 0), ('graphInfoV1', QnnSystemContext_GraphInfoV1_t, 8), ('graphInfoV2', QnnSystemContext_GraphInfoV2_t, 8), ('graphInfoV3', QnnSystemContext_GraphInfoV3_t, 8)])
QnnSystemContext_BinaryInfoV1_t.register_fields([('backendId', uint32_t, 0), ('buildId', c.POINTER[ctypes.c_char], 8), ('coreApiVersion', Qnn_Version_t, 16), ('backendApiVersion', Qnn_Version_t, 28), ('socVersion', c.POINTER[ctypes.c_char], 40), ('hwInfoBlobVersion', Qnn_Version_t, 48), ('contextBlobVersion', Qnn_Version_t, 60), ('hwInfoBlobSize', uint32_t, 72), ('hwInfoBlob', ctypes.c_void_p, 80), ('contextBlobSize', uint64_t, 88), ('numContextTensors', uint32_t, 96), ('contextTensors', c.POINTER[Qnn_Tensor_t], 104), ('numGraphs', uint32_t, 112), ('graphs', c.POINTER[QnnSystemContext_GraphInfo_t], 120)])
@c.record
class QnnSystemContext_BinaryInfoV2_t(c.Struct):
  SIZE = 136
  backendId: int
  buildId: c.POINTER[ctypes.c_char]
  coreApiVersion: Qnn_Version_t
  backendApiVersion: Qnn_Version_t
  socVersion: c.POINTER[ctypes.c_char]
  hwInfoBlobVersion: Qnn_Version_t
  contextBlobVersion: Qnn_Version_t
  hwInfoBlobSize: int
  hwInfoBlob: ctypes.c_void_p
  contextBlobSize: int
  numContextTensors: int
  contextTensors: c.POINTER[Qnn_Tensor_t]
  numGraphs: int
  graphs: c.POINTER[QnnSystemContext_GraphInfo_t]
  platformInfo: c.POINTER[QnnDevice_PlatformInfo_t]
QnnSystemContext_BinaryInfoV2_t.register_fields([('backendId', uint32_t, 0), ('buildId', c.POINTER[ctypes.c_char], 8), ('coreApiVersion', Qnn_Version_t, 16), ('backendApiVersion', Qnn_Version_t, 28), ('socVersion', c.POINTER[ctypes.c_char], 40), ('hwInfoBlobVersion', Qnn_Version_t, 48), ('contextBlobVersion', Qnn_Version_t, 60), ('hwInfoBlobSize', uint32_t, 72), ('hwInfoBlob', ctypes.c_void_p, 80), ('contextBlobSize', uint64_t, 88), ('numContextTensors', uint32_t, 96), ('contextTensors', c.POINTER[Qnn_Tensor_t], 104), ('numGraphs', uint32_t, 112), ('graphs', c.POINTER[QnnSystemContext_GraphInfo_t], 120), ('platformInfo', c.POINTER[QnnDevice_PlatformInfo_t], 128)])
@c.record
class QnnSystemContext_BinaryInfoV3_t(c.Struct):
  SIZE = 136
  backendId: int
  buildId: c.POINTER[ctypes.c_char]
  coreApiVersion: Qnn_Version_t
  backendApiVersion: Qnn_Version_t
  socVersion: c.POINTER[ctypes.c_char]
  contextBlobVersion: Qnn_Version_t
  contextBlobSize: int
  numContextTensors: int
  contextTensors: c.POINTER[Qnn_Tensor_t]
  numGraphs: int
  graphs: c.POINTER[QnnSystemContext_GraphInfo_t]
  platformInfo: c.POINTER[QnnDevice_PlatformInfo_t]
  contextMetadataSize: int
  contextMetadata: ctypes.c_void_p
  socModel: int
QnnSystemContext_BinaryInfoV3_t.register_fields([('backendId', uint32_t, 0), ('buildId', c.POINTER[ctypes.c_char], 8), ('coreApiVersion', Qnn_Version_t, 16), ('backendApiVersion', Qnn_Version_t, 28), ('socVersion', c.POINTER[ctypes.c_char], 40), ('contextBlobVersion', Qnn_Version_t, 48), ('contextBlobSize', uint64_t, 64), ('numContextTensors', uint32_t, 72), ('contextTensors', c.POINTER[Qnn_Tensor_t], 80), ('numGraphs', uint32_t, 88), ('graphs', c.POINTER[QnnSystemContext_GraphInfo_t], 96), ('platformInfo', c.POINTER[QnnDevice_PlatformInfo_t], 104), ('contextMetadataSize', uint32_t, 112), ('contextMetadata', ctypes.c_void_p, 120), ('socModel', uint32_t, 128)])
QnnSystemContext_BinaryInfo_t.register_fields([('version', ctypes.c_uint32, 0), ('contextBinaryInfoV1', QnnSystemContext_BinaryInfoV1_t, 8), ('contextBinaryInfoV2', QnnSystemContext_BinaryInfoV2_t, 8), ('contextBinaryInfoV3', QnnSystemContext_BinaryInfoV3_t, 8)])
QnnSystemContext_GetBinaryInfoFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, c.POINTER[c.POINTER[QnnSystemContext_BinaryInfo_t]], c.POINTER[ctypes.c_uint64]]]
QnnSystemContext_GetMetaDataFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, c.POINTER[c.POINTER[QnnSystemContext_BinaryInfo_t]]]]
QnnSystemContext_FreeFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p]]
QnnSystemTensor_getMemoryFootprintFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [Qnn_Tensor_t, c.POINTER[ctypes.c_uint64]]]
QnnSystemLog_createFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [c.CFUNCTYPE[None, [c.POINTER[ctypes.c_char], ctypes.c_uint32, ctypes.c_uint64, c.POINTER[struct___va_list_tag]]], ctypes.c_uint32, c.POINTER[ctypes.c_void_p]]]
QnnSystemLog_setLogLevelFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_uint32]]
QnnSystemLog_freeFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p]]
QnnSystemDlc_createFromFileFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[ctypes.c_char], c.POINTER[ctypes.c_void_p]]]
QnnSystemDlc_createFromBinaryFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[ctypes.c_ubyte], ctypes.c_uint64, c.POINTER[ctypes.c_void_p]]]
@c.record
class QnnSystemDlc_GraphConfigInfo_t(c.Struct):
  SIZE = 32
  version: int
  v1: QnnSystemDlc_GraphConfigInfoV1_t
QnnSystemContext_GraphConfigInfoVersion_t: dict[int, str] = {(QNN_SYSTEM_DLC_GRAPH_CONFIG_INFO_VERSION_1:=1): 'QNN_SYSTEM_DLC_GRAPH_CONFIG_INFO_VERSION_1', (QNN_SYSTEM_DLC_GRAPH_CONFIG_INFO_UNDEFINED:=2147483647): 'QNN_SYSTEM_DLC_GRAPH_CONFIG_INFO_UNDEFINED'}
@c.record
class QnnSystemDlc_GraphConfigInfoV1_t(c.Struct):
  SIZE = 24
  graphName: c.POINTER[ctypes.c_char]
  graphConfigs: c.POINTER[c.POINTER[QnnGraph_Config_t]]
  numConfigs: int
QnnSystemDlc_GraphConfigInfoV1_t.register_fields([('graphName', c.POINTER[ctypes.c_char], 0), ('graphConfigs', c.POINTER[c.POINTER[QnnGraph_Config_t]], 8), ('numConfigs', uint32_t, 16)])
QnnSystemDlc_GraphConfigInfo_t.register_fields([('version', ctypes.c_uint32, 0), ('v1', QnnSystemDlc_GraphConfigInfoV1_t, 8)])
QnnSystemDlc_composeGraphsFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnSystemDlc_GraphConfigInfo_t]], ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, QnnInterface_t, ctypes.c_uint32, c.POINTER[c.POINTER[QnnSystemContext_GraphInfo_t]], c.POINTER[ctypes.c_uint32]]]
@c.record
class Qnn_OpMapping_t(c.Struct):
  SIZE = 16
  version: int
  v1: c.POINTER[Qnn_OpMappingV1_t]
Qnn_OpMappingVersion_t: dict[int, str] = {(QNN_OP_MAPPING_VERSION_1:=1): 'QNN_OP_MAPPING_VERSION_1'}
@c.record
class Qnn_OpMappingV1_t(c.Struct):
  SIZE = 40
  graphName: c.POINTER[ctypes.c_char]
  opMappings: c.POINTER[Qnn_Mapping_t]
  numOpMappings: int
  tensorMappings: c.POINTER[Qnn_Mapping_t]
  numTensorMappings: int
@c.record
class Qnn_Mapping_t(c.Struct):
  SIZE = 24
  name: c.POINTER[ctypes.c_char]
  pair: c.POINTER[Qnn_OpMappingPair_t]
  numPairs: int
@c.record
class Qnn_OpMappingPair_t(c.Struct):
  SIZE = 16
  name: c.POINTER[ctypes.c_char]
  type: int
Qnn_MappingType_t: dict[int, str] = {(QNN_OP_MAPPING_TYPE_TENSOR:=0): 'QNN_OP_MAPPING_TYPE_TENSOR', (QNN_OP_MAPPING_TYPE_OP:=1): 'QNN_OP_MAPPING_TYPE_OP'}
Qnn_OpMappingPair_t.register_fields([('name', c.POINTER[ctypes.c_char], 0), ('type', ctypes.c_uint32, 8)])
Qnn_Mapping_t.register_fields([('name', c.POINTER[ctypes.c_char], 0), ('pair', c.POINTER[Qnn_OpMappingPair_t], 8), ('numPairs', uint32_t, 16)])
Qnn_OpMappingV1_t.register_fields([('graphName', c.POINTER[ctypes.c_char], 0), ('opMappings', c.POINTER[Qnn_Mapping_t], 8), ('numOpMappings', uint32_t, 16), ('tensorMappings', c.POINTER[Qnn_Mapping_t], 24), ('numTensorMappings', uint32_t, 32)])
Qnn_OpMapping_t.register_fields([('version', ctypes.c_uint32, 0), ('v1', c.POINTER[Qnn_OpMappingV1_t], 8)])
QnnSystemDlc_getOpMappingsFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[Qnn_OpMapping_t]], c.POINTER[ctypes.c_uint32]]]
QnnSystemDlc_freeFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p]]
@c.record
class QnnSystemProfile_SerializationTarget_t(c.Struct):
  SIZE = 24
  type: int
  file: QnnSystemProfile_SerializationTargetFile_t
QnnSystemProfile_SerializationTargetType_t: dict[int, str] = {(QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_FILE:=1): 'QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_FILE', (QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_UNDEFINED:=2147483647): 'QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_UNDEFINED'}
@c.record
class QnnSystemProfile_SerializationTargetFile_t(c.Struct):
  SIZE = 16
  fileName: c.POINTER[ctypes.c_char]
  fileDirectory: c.POINTER[ctypes.c_char]
QnnSystemProfile_SerializationTargetFile_t.register_fields([('fileName', c.POINTER[ctypes.c_char], 0), ('fileDirectory', c.POINTER[ctypes.c_char], 8)])
QnnSystemProfile_SerializationTarget_t.register_fields([('type', ctypes.c_uint32, 0), ('file', QnnSystemProfile_SerializationTargetFile_t, 8)])
@c.record
class QnnSystemProfile_SerializationTargetConfig_t(c.Struct):
  SIZE = 32
  type: int
  maxNumMessages: int
  serializationHeader: QnnSystemProfile_SerializationFileHeader_t
QnnSystemProfile_SerializationTargetConfigType_t: dict[int, str] = {(QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_CONFIG_MAX_NUM_MESSAGES:=0): 'QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_CONFIG_MAX_NUM_MESSAGES', (QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_CONFIG_SERIALIZATION_HEADER:=1): 'QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_CONFIG_SERIALIZATION_HEADER', (QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_CONFIG_UNDEFINED:=2147483647): 'QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_CONFIG_UNDEFINED'}
@c.record
class QnnSystemProfile_SerializationFileHeader_t(c.Struct):
  SIZE = 24
  appName: c.POINTER[ctypes.c_char]
  appVersion: c.POINTER[ctypes.c_char]
  backendVersion: c.POINTER[ctypes.c_char]
QnnSystemProfile_SerializationFileHeader_t.register_fields([('appName', c.POINTER[ctypes.c_char], 0), ('appVersion', c.POINTER[ctypes.c_char], 8), ('backendVersion', c.POINTER[ctypes.c_char], 16)])
QnnSystemProfile_SerializationTargetConfig_t.register_fields([('type', ctypes.c_uint32, 0), ('maxNumMessages', uint32_t, 8), ('serializationHeader', QnnSystemProfile_SerializationFileHeader_t, 8)])
QnnSystemProfile_createSerializationTargetFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [QnnSystemProfile_SerializationTarget_t, c.POINTER[QnnSystemProfile_SerializationTargetConfig_t], ctypes.c_uint32, c.POINTER[ctypes.c_void_p]]]
@c.record
class QnnSystemProfile_ProfileData_t(c.Struct):
  SIZE = 72
  version: int
  v1: QnnSystemProfile_ProfileDataV1_t
QnnSystemProfile_ProfileDataVersion_t: dict[int, str] = {(QNN_SYSTEM_PROFILE_DATA_VERSION_1:=1): 'QNN_SYSTEM_PROFILE_DATA_VERSION_1', (QNN_SYSTEM_PROFILE_DATA_VERSION_UNDEFINED:=2147483647): 'QNN_SYSTEM_PROFILE_DATA_VERSION_UNDEFINED'}
@c.record
class QnnSystemProfile_ProfileDataV1_t(c.Struct):
  SIZE = 64
  header: QnnSystemProfile_HeaderV1_t
  profilingEvents: c.POINTER[struct_QnnSystemProfile_ProfileEventV1_t]
  numProfilingEvents: int
@c.record
class QnnSystemProfile_HeaderV1_t(c.Struct):
  SIZE = 48
  startTime: int
  stopTime: int
  startMem: int
  stopMem: int
  methodType: int
  visibility: int
  graphName: c.POINTER[ctypes.c_char]
QnnSystemProfile_MethodType_t: dict[int, str] = {(QNN_SYSTEM_PROFILE_METHOD_TYPE_NONE:=0): 'QNN_SYSTEM_PROFILE_METHOD_TYPE_NONE', (QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_EXECUTE:=1): 'QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_EXECUTE', (QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_FINALIZE:=2): 'QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_FINALIZE', (QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_EXECUTE_ASYNC:=3): 'QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_EXECUTE_ASYNC', (QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_CREATE_FROM_BINARY:=4): 'QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_CREATE_FROM_BINARY', (QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_DEINIT:=5): 'QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_DEINIT', (QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_CONTEXT_CREATE:=6): 'QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_CONTEXT_CREATE', (QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_COMPOSE_GRAPHS:=7): 'QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_COMPOSE_GRAPHS', (QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_EXECUTE_IPS:=8): 'QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_EXECUTE_IPS', (QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_GRAPH_COMPONENT:=9): 'QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_GRAPH_COMPONENT', (QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_BACKEND_LIB_LOAD:=10): 'QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_BACKEND_LIB_LOAD', (QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_APPLY_BINARY_SECTION:=11): 'QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_APPLY_BINARY_SECTION', (QNN_SYSTEM_PROFILE_METHOD_TYPE_CONTEXT_FINALIZE:=12): 'QNN_SYSTEM_PROFILE_METHOD_TYPE_CONTEXT_FINALIZE', (QNN_SYSTEM_PROFILE_METHOD_TYPE_CONTEXT_GET_BINARY_SIZE:=13): 'QNN_SYSTEM_PROFILE_METHOD_TYPE_CONTEXT_GET_BINARY_SIZE', (QNN_SYSTEM_PROFILE_METHOD_TYPE_CONTEXT_GET_BINARY:=14): 'QNN_SYSTEM_PROFILE_METHOD_TYPE_CONTEXT_GET_BINARY', (QNN_SYSTEM_PROFILE_METHOD_TYPE_CONTEXT_GET_BINARY_SECTION_SIZE:=15): 'QNN_SYSTEM_PROFILE_METHOD_TYPE_CONTEXT_GET_BINARY_SECTION_SIZE', (QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_FINALIZE_TENSOR_UPDATES:=16): 'QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_FINALIZE_TENSOR_UPDATES'}
QnnSystemProfile_Visibility_t: dict[int, str] = {(QNN_SYSTEM_PROFILE_VISIBILITY_PUBLIC:=0): 'QNN_SYSTEM_PROFILE_VISIBILITY_PUBLIC', (QNN_SYSTEM_PROFILE_VISIBILITY_PRIVATE:=1): 'QNN_SYSTEM_PROFILE_VISIBILITY_PRIVATE'}
QnnSystemProfile_HeaderV1_t.register_fields([('startTime', uint64_t, 0), ('stopTime', uint64_t, 8), ('startMem', uint64_t, 16), ('stopMem', uint64_t, 24), ('methodType', ctypes.c_uint32, 32), ('visibility', ctypes.c_uint32, 36), ('graphName', c.POINTER[ctypes.c_char], 40)])
@c.record
class struct_QnnSystemProfile_ProfileEventV1_t(c.Struct):
  SIZE = 80
  type: int
  eventData: QnnProfile_EventData_t
  extendedEventData: QnnProfile_ExtendedEventData_t
  profileSubEventData: c.POINTER[struct_QnnSystemProfile_ProfileEventV1_t]
  numSubEvents: int
QnnSystemProfile_ProfileEventV1_t: TypeAlias = struct_QnnSystemProfile_ProfileEventV1_t
QnnSystemProfile_EventDataType_t: dict[int, str] = {(QNN_SYSTEM_PROFILE_EVENT_DATA:=0): 'QNN_SYSTEM_PROFILE_EVENT_DATA', (QNN_SYSTEM_PROFILE_EXTENDED_EVENT_DATA:=1): 'QNN_SYSTEM_PROFILE_EXTENDED_EVENT_DATA', (QNN_SYSTEM_PROFILE_EVENT_DATA_UNDEFINED:=2147483647): 'QNN_SYSTEM_PROFILE_EVENT_DATA_UNDEFINED'}
struct_QnnSystemProfile_ProfileEventV1_t.register_fields([('type', ctypes.c_uint32, 0), ('eventData', QnnProfile_EventData_t, 8), ('extendedEventData', QnnProfile_ExtendedEventData_t, 8), ('profileSubEventData', c.POINTER[QnnSystemProfile_ProfileEventV1_t], 64), ('numSubEvents', uint32_t, 72)])
QnnSystemProfile_ProfileDataV1_t.register_fields([('header', QnnSystemProfile_HeaderV1_t, 0), ('profilingEvents', c.POINTER[QnnSystemProfile_ProfileEventV1_t], 48), ('numProfilingEvents', uint32_t, 56)])
QnnSystemProfile_ProfileData_t.register_fields([('version', ctypes.c_uint32, 0), ('v1', QnnSystemProfile_ProfileDataV1_t, 8)])
QnnSystemProfile_serializeEventDataFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p, c.POINTER[c.POINTER[QnnSystemProfile_ProfileData_t]], ctypes.c_uint32]]
QnnSystemProfile_freeSerializationTargetFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_void_p]]
QnnSystemInterface_ImplementationV1_7_t.register_fields([('systemContextCreate', QnnSystemContext_CreateFn_t, 0), ('systemContextGetBinaryInfo', QnnSystemContext_GetBinaryInfoFn_t, 8), ('systemContextGetMetaData', QnnSystemContext_GetMetaDataFn_t, 16), ('systemContextFree', QnnSystemContext_FreeFn_t, 24), ('systemTensorGetMemoryFootprint', QnnSystemTensor_getMemoryFootprintFn_t, 32), ('systemLogCreate', QnnSystemLog_createFn_t, 40), ('systemLogSetLogLevel', QnnSystemLog_setLogLevelFn_t, 48), ('systemLogFree', QnnSystemLog_freeFn_t, 56), ('systemDlcCreateFromFile', QnnSystemDlc_createFromFileFn_t, 64), ('systemDlcCreateFromBinary', QnnSystemDlc_createFromBinaryFn_t, 72), ('systemDlcComposeGraphs', QnnSystemDlc_composeGraphsFn_t, 80), ('systemDlcGetOpMappings', QnnSystemDlc_getOpMappingsFn_t, 88), ('systemDlcFree', QnnSystemDlc_freeFn_t, 96), ('systemProfileCreateSerializationTarget', QnnSystemProfile_createSerializationTargetFn_t, 104), ('systemProfileSerializeEventData', QnnSystemProfile_serializeEventDataFn_t, 112), ('systemProfileFreeSerializationTarget', QnnSystemProfile_freeSerializationTargetFn_t, 120)])
QnnSystemInterface_t.register_fields([('backendId', uint32_t, 0), ('providerName', c.POINTER[ctypes.c_char], 8), ('systemApiVersion', Qnn_Version_t, 16), ('v1_7', QnnSystemInterface_ImplementationV1_7_t, 32)])
QnnHtpDevice_InfrastructureType_t: dict[int, str] = {(QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF:=0): 'QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF', (QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_UNKNOWN:=2147483647): 'QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_UNKNOWN'}
@c.record
class QnnHtpDevice_PerfInfrastructure_t(c.Struct):
  SIZE = 32
  createPowerConfigId: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_uint32, ctypes.c_uint32, c.POINTER[ctypes.c_uint32]]]
  destroyPowerConfigId: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_uint32]]
  setPowerConfig: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_uint32, c.POINTER[c.POINTER[QnnHtpPerfInfrastructure_PowerConfig_t]]]]
  setMemoryConfig: c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_uint32, ctypes.c_uint32, c.POINTER[c.POINTER[QnnHtpPerfInfrastructure_MemoryConfig_t]]]]
QnnHtpPerfInfrastructure_CreatePowerConfigIdFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_uint32, ctypes.c_uint32, c.POINTER[ctypes.c_uint32]]]
QnnHtpPerfInfrastructure_DestroyPowerConfigIdFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_uint32]]
@c.record
class QnnHtpPerfInfrastructure_PowerConfig_t(c.Struct):
  SIZE = 68
  option: int
  dcvsV3Config: QnnHtpPerfInfrastructure_DcvsV3_t
  rpcControlLatencyConfig: int
  rpcPollingTimeConfig: int
  hmxTimeoutIntervalUsConfig: int
  hmxV2Config: QnnHtpPerfInfrastructure_HmxV2_t
  adaptivePollingTimeConfig: int
  ddrPerfModeConfig: int
QnnHtpPerfInfrastructure_PowerConfigOption_t: dict[int, str] = {(QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3:=1): 'QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3', (QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY:=2): 'QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY', (QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME:=3): 'QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME', (QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_TIMEOUT_INTERVAL_US:=4): 'QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_TIMEOUT_INTERVAL_US', (QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_V2:=5): 'QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_V2', (QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_ADAPTIVE_POLLING_TIME:=6): 'QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_ADAPTIVE_POLLING_TIME', (QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DDR_PERF_MODE:=7): 'QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DDR_PERF_MODE', (QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_UNKNOWN:=2147483647): 'QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_UNKNOWN'}
@c.record
class QnnHtpPerfInfrastructure_DcvsV3_t(c.Struct):
  SIZE = 64
  contextId: int
  setDcvsEnable: int
  dcvsEnable: int
  powerMode: int
  setSleepLatency: int
  sleepLatency: int
  setSleepDisable: int
  sleepDisable: int
  setBusParams: int
  busVoltageCornerMin: int
  busVoltageCornerTarget: int
  busVoltageCornerMax: int
  setCoreParams: int
  coreVoltageCornerMin: int
  coreVoltageCornerTarget: int
  coreVoltageCornerMax: int
QnnHtpPerfInfrastructure_SetDcvsEnable_t: TypeAlias = ctypes.c_uint32
QnnHtpPerfInfrastructure_DcvsEnable_t: TypeAlias = ctypes.c_uint32
QnnHtpPerfInfrastructure_PowerMode_t: dict[int, str] = {(QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN:=1): 'QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN', (QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_ONLY_UP:=2): 'QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_ONLY_UP', (QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE:=4): 'QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE', (QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_AGGRESSIVE_MODE:=8): 'QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_AGGRESSIVE_MODE', (QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE:=16): 'QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE', (QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_DUTY_CYCLE_MODE:=32): 'QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_DUTY_CYCLE_MODE', (QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_UNKNOWN:=2147483647): 'QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_UNKNOWN'}
QnnHtpPerfInfrastructure_SetSleepLatency_t: TypeAlias = ctypes.c_uint32
QnnHtpPerfInfrastructure_SleepLatency_t: TypeAlias = ctypes.c_uint32
QnnHtpPerfInfrastructure_SetSleepDisable_t: TypeAlias = ctypes.c_uint32
QnnHtpPerfInfrastructure_SleepDisable_t: TypeAlias = ctypes.c_uint32
QnnHtpPerfInfrastructure_SetBusParams_t: TypeAlias = ctypes.c_uint32
QnnHtpPerfInfrastructure_VoltageCorner_t: dict[int, str] = {(DCVS_VOLTAGE_CORNER_DISABLE:=16): 'DCVS_VOLTAGE_CORNER_DISABLE', (DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER:=32): 'DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER', (DCVS_VOLTAGE_VCORNER_SVS2:=48): 'DCVS_VOLTAGE_VCORNER_SVS2', (DCVS_VOLTAGE_VCORNER_SVS:=64): 'DCVS_VOLTAGE_VCORNER_SVS', (DCVS_VOLTAGE_VCORNER_SVS_PLUS:=80): 'DCVS_VOLTAGE_VCORNER_SVS_PLUS', (DCVS_VOLTAGE_VCORNER_NOM:=96): 'DCVS_VOLTAGE_VCORNER_NOM', (DCVS_VOLTAGE_VCORNER_NOM_PLUS:=112): 'DCVS_VOLTAGE_VCORNER_NOM_PLUS', (DCVS_VOLTAGE_VCORNER_TURBO:=128): 'DCVS_VOLTAGE_VCORNER_TURBO', (DCVS_VOLTAGE_VCORNER_TURBO_PLUS:=144): 'DCVS_VOLTAGE_VCORNER_TURBO_PLUS', (DCVS_VOLTAGE_VCORNER_TURBO_L2:=146): 'DCVS_VOLTAGE_VCORNER_TURBO_L2', (DCVS_VOLTAGE_VCORNER_TURBO_L3:=147): 'DCVS_VOLTAGE_VCORNER_TURBO_L3', (DCVS_VOLTAGE_VCORNER_TURBO_L4:=148): 'DCVS_VOLTAGE_VCORNER_TURBO_L4', (DCVS_VOLTAGE_VCORNER_TURBO_L5:=149): 'DCVS_VOLTAGE_VCORNER_TURBO_L5', (DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER:=160): 'DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER', (DCVS_VOLTAGE_VCORNER_UNKNOWN:=2147483647): 'DCVS_VOLTAGE_VCORNER_UNKNOWN'}
QnnHtpPerfInfrastructure_SetCoreParams_t: TypeAlias = ctypes.c_uint32
QnnHtpPerfInfrastructure_DcvsV3_t.register_fields([('contextId', uint32_t, 0), ('setDcvsEnable', QnnHtpPerfInfrastructure_SetDcvsEnable_t, 4), ('dcvsEnable', QnnHtpPerfInfrastructure_DcvsEnable_t, 8), ('powerMode', ctypes.c_uint32, 12), ('setSleepLatency', QnnHtpPerfInfrastructure_SetSleepLatency_t, 16), ('sleepLatency', QnnHtpPerfInfrastructure_SleepLatency_t, 20), ('setSleepDisable', QnnHtpPerfInfrastructure_SetSleepDisable_t, 24), ('sleepDisable', QnnHtpPerfInfrastructure_SleepDisable_t, 28), ('setBusParams', QnnHtpPerfInfrastructure_SetBusParams_t, 32), ('busVoltageCornerMin', ctypes.c_uint32, 36), ('busVoltageCornerTarget', ctypes.c_uint32, 40), ('busVoltageCornerMax', ctypes.c_uint32, 44), ('setCoreParams', QnnHtpPerfInfrastructure_SetCoreParams_t, 48), ('coreVoltageCornerMin', ctypes.c_uint32, 52), ('coreVoltageCornerTarget', ctypes.c_uint32, 56), ('coreVoltageCornerMax', ctypes.c_uint32, 60)])
QnnHtpPerfInfrastructure_RpcControlLatency_t: TypeAlias = ctypes.c_uint32
QnnHtpPerfInfrastructure_RpcPollingTime_t: TypeAlias = ctypes.c_uint32
QnnHtpPerfInfrastructure_HmxTimeoutIntervalUs_t: TypeAlias = ctypes.c_uint32
@c.record
class QnnHtpPerfInfrastructure_HmxV2_t(c.Struct):
  SIZE = 20
  hmxPickDefault: int
  hmxVoltageCornerMin: int
  hmxVoltageCornerTarget: int
  hmxVoltageCornerMax: int
  hmxPerfMode: int
QnnHtpPerfInfrastructure_HmxDefault_Vote_t: TypeAlias = ctypes.c_uint32
QnnHtpPerfInfrastructure_ExpVoltageCorner_t: dict[int, str] = {(DCVS_EXP_VCORNER_DISABLE:=0): 'DCVS_EXP_VCORNER_DISABLE', (DCVS_EXP_VCORNER_MIN:=256): 'DCVS_EXP_VCORNER_MIN', (DCVS_EXP_VCORNER_LOW_SVS_D2:=308): 'DCVS_EXP_VCORNER_LOW_SVS_D2', (DCVS_EXP_VCORNER_LOW_SVS_D1:=312): 'DCVS_EXP_VCORNER_LOW_SVS_D1', (DCVS_EXP_VCORNER_LOW_SVS:=320): 'DCVS_EXP_VCORNER_LOW_SVS', (DCVS_EXP_VCORNER_LOW_SVS_L0:=332): 'DCVS_EXP_VCORNER_LOW_SVS_L0', (DCVS_EXP_VCORNER_SVS:=384): 'DCVS_EXP_VCORNER_SVS', (DCVS_EXP_VCORNER_SVS_L0:=400): 'DCVS_EXP_VCORNER_SVS_L0', (DCVS_EXP_VCORNER_SVS_L1:=448): 'DCVS_EXP_VCORNER_SVS_L1', (DCVS_EXP_VCORNER_SVS_L2:=480): 'DCVS_EXP_VCORNER_SVS_L2', (DCVS_EXP_VCORNER_NOM:=512): 'DCVS_EXP_VCORNER_NOM', (DCVS_EXP_VCORNER_NOM_L1:=576): 'DCVS_EXP_VCORNER_NOM_L1', (DCVS_EXP_VCORNER_NOM_L2:=592): 'DCVS_EXP_VCORNER_NOM_L2', (DCVS_EXP_VCORNER_TUR:=640): 'DCVS_EXP_VCORNER_TUR', (DCVS_EXP_VCORNER_TUR_L1:=672): 'DCVS_EXP_VCORNER_TUR_L1', (DCVS_EXP_VCORNER_TUR_L2:=688): 'DCVS_EXP_VCORNER_TUR_L2', (DCVS_EXP_VCORNER_TUR_L3:=704): 'DCVS_EXP_VCORNER_TUR_L3', (DCVS_EXP_VCORNER_MAX:=65535): 'DCVS_EXP_VCORNER_MAX', (DCVS_EXP_VCORNER_UNKNOWN:=2147483647): 'DCVS_EXP_VCORNER_UNKNOWN'}
QnnHtpPerfInfrastructure_ClkPerfMode_t: dict[int, str] = {(QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_HIGH:=0): 'QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_HIGH', (QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_LOW:=1): 'QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_LOW', (QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_UNKNOWN:=2147483647): 'QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_UNKNOWN'}
QnnHtpPerfInfrastructure_HmxV2_t.register_fields([('hmxPickDefault', QnnHtpPerfInfrastructure_HmxDefault_Vote_t, 0), ('hmxVoltageCornerMin', ctypes.c_uint32, 4), ('hmxVoltageCornerTarget', ctypes.c_uint32, 8), ('hmxVoltageCornerMax', ctypes.c_uint32, 12), ('hmxPerfMode', ctypes.c_uint32, 16)])
QnnHtpPerfInfrastructure_AdaptivePollingTime_t: TypeAlias = ctypes.c_uint32
QnnHtpPerfInfrastructure_DdrPerfMode_t: TypeAlias = ctypes.c_uint32
QnnHtpPerfInfrastructure_PowerConfig_t.register_fields([('option', ctypes.c_uint32, 0), ('dcvsV3Config', QnnHtpPerfInfrastructure_DcvsV3_t, 4), ('rpcControlLatencyConfig', QnnHtpPerfInfrastructure_RpcControlLatency_t, 4), ('rpcPollingTimeConfig', QnnHtpPerfInfrastructure_RpcPollingTime_t, 4), ('hmxTimeoutIntervalUsConfig', QnnHtpPerfInfrastructure_HmxTimeoutIntervalUs_t, 4), ('hmxV2Config', QnnHtpPerfInfrastructure_HmxV2_t, 4), ('adaptivePollingTimeConfig', QnnHtpPerfInfrastructure_AdaptivePollingTime_t, 4), ('ddrPerfModeConfig', QnnHtpPerfInfrastructure_DdrPerfMode_t, 4)])
QnnHtpPerfInfrastructure_SetPowerConfigFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_uint32, c.POINTER[c.POINTER[QnnHtpPerfInfrastructure_PowerConfig_t]]]]
@c.record
class QnnHtpPerfInfrastructure_MemoryConfig_t(c.Struct):
  SIZE = 8
  option: int
  memGrowSizeConfig: int
QnnHtpPerfInfrastructure_MemoryConfigOption_t: dict[int, str] = {(QNN_HTP_PERF_INFRASTRUCTURE_MEMORY_CONFIGOPTION_GROW_SIZE:=1): 'QNN_HTP_PERF_INFRASTRUCTURE_MEMORY_CONFIGOPTION_GROW_SIZE', (QNN_HTP_PERF_INFRASTRUCTURE_MEMORY_CONFIGOPTION_UNKNOWN:=2147483647): 'QNN_HTP_PERF_INFRASTRUCTURE_MEMORY_CONFIGOPTION_UNKNOWN'}
QnnHtpPerfInfrastructure_MemGrowSize_t: TypeAlias = ctypes.c_uint32
QnnHtpPerfInfrastructure_MemoryConfig_t.register_fields([('option', ctypes.c_uint32, 0), ('memGrowSizeConfig', QnnHtpPerfInfrastructure_MemGrowSize_t, 4)])
QnnHtpPerfInfrastructure_SetMemoryConfigFn_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [ctypes.c_uint32, ctypes.c_uint32, c.POINTER[c.POINTER[QnnHtpPerfInfrastructure_MemoryConfig_t]]]]
QnnHtpDevice_PerfInfrastructure_t.register_fields([('createPowerConfigId', QnnHtpPerfInfrastructure_CreatePowerConfigIdFn_t, 0), ('destroyPowerConfigId', QnnHtpPerfInfrastructure_DestroyPowerConfigIdFn_t, 8), ('setPowerConfig', QnnHtpPerfInfrastructure_SetPowerConfigFn_t, 16), ('setMemoryConfig', QnnHtpPerfInfrastructure_SetMemoryConfigFn_t, 24)])
struct__QnnDevice_Infrastructure_t.register_fields([('infraType', ctypes.c_uint32, 0), ('perfInfra', QnnHtpDevice_PerfInfrastructure_t, 8)])
QnnHtpDevice_Infrastructure_t: TypeAlias = struct__QnnDevice_Infrastructure_t
