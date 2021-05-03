# Lint as: python3
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging

import numpy as np

from .binding import HalDevice, HalElementType, VmContext, VmFunction, VmVariantList

__all__ = [
    "FunctionInvoker",
]


class Invocation:

  def __init__(self, device: HalDevice):
    self.device = device


class FunctionInvoker:
  """Wraps a VmFunction, enabling invocations against it."""

  def __init__(self, vm_context: VmContext, device: HalDevice,
               vm_function: VmFunction):
    self._vm_context = vm_context
    # TODO: Needing to know the precise device to allocate on here is bad
    # layering and will need to be fixed in some fashion if/when doing
    # heterogenous dispatch.
    self._device = device
    self._vm_function = vm_function
    self._abi_dict = None
    self._arg_descs = None
    self._ret_descs = None
    self._parse_abi_dict(vm_function)

  @property
  def vm_function(self) -> VmFunction:
    return self._vm_function

  def __call__(self, *args):
    # Initialize the capacity to our total number of args, since we should
    # be below that when doing a flat invocation. May want to be more
    # conservative here when considering nesting.
    inv = Invocation(self._device)
    ret_descs = self._ret_descs
    arg_list = VmVariantList(len(args))
    ret_list = VmVariantList(len(ret_descs) if ret_descs is not None else 1)
    _merge_python_sequence_to_vm(inv, arg_list, args, self._arg_descs)
    self._vm_context.invoke(self._vm_function, arg_list, ret_list)
    returns = _extract_vm_sequence_to_python(inv, ret_list, ret_descs)
    return_arity = len(returns)
    if return_arity == 1:
      return returns[0]
    elif return_arity == 0:
      return None
    else:
      return tuple(returns)

  def _parse_abi_dict(self, vm_function: VmFunction):
    reflection = vm_function.reflection
    abi_json = reflection.get("iree.abi")
    if abi_json is None:
      # It is valid to have no reflection data, and rely on pure dynamic
      # dispatch.
      logging.warning(
          "Function lacks reflection data. Interop will be limited: %r",
          vm_function)
      return
    try:
      self._abi_dict = json.loads(abi_json)
    except json.JSONDecodeError as e:
      raise RuntimeError(
          f"Reflection metadata is not valid JSON: {abi_json}") from e
    try:
      self._arg_descs = self._abi_dict["a"]
      self._ret_descs = self._abi_dict["r"]
    except KeyError as e:
      raise RuntimeError(
          f"Malformed function reflection metadata: {reflection}") from e
    if not isinstance(self._arg_descs, list) or not isinstance(
        self._ret_descs, list):
      raise RuntimeError(
          f"Malformed function reflection metadata structure: {reflection}")

  def __repr__(self):
    return repr(self._vm_function)


# Python type to VM Type converters. All of these take:
#   inv: Invocation
#   target_list: VmVariantList to append to
#   python_value: The python value of the given type
#   desc: The ABI descriptor list (or None if in dynamic mode).


def _bool_to_vm(inv: Invocation, t: VmVariantList, x, desc):
  _int_to_vm(inv, t, int(x), desc)


def _int_to_vm(inv: Invocation, t: VmVariantList, x, desc):
  _raise_argument_error(inv, "Python int arguments not yet supported")


def _float_to_vm(inv: Invocation, t: VmVariantList, x, desc):
  _raise_argument_error(inv, "Python float arguments not yet supported")


def _list_to_vm(inv: Invocation, t: VmVariantList, x, desc):
  _raise_argument_error(inv, "Python list arguments not yet supported")


def _tuple_to_vm(inv: Invocation, t: VmVariantList, x, desc):
  _raise_argument_error(inv, "Python tuple arguments not yet supported")


def _dict_to_vm(inv: Invocation, t: VmVariantList, x, desc):
  _raise_argument_error(inv, "Python dict arguments not yet supported")


def _str_to_vm(inv: Invocation, t: VmVariantList, x, desc):
  _raise_argument_error(inv, "Python str arguments not yet supported")


def _ndarray_to_vm(inv: Invocation, t: VmVariantList, x, desc):
  # Validate and implicit conversion against type descriptor.
  if desc is not None:
    desc_type = desc[0]
    if desc_type != "ndarray":
      _raise_argument_error(inv, f"passed an ndarray but expected {desc_type}")
    dtype_str = desc[1]
    try:
      dtype = ABI_TYPE_TO_DTYPE[dtype_str]
    except KeyError:
      _raise_argument_error(inv, f"unrecognized dtype '{dtype_str}'")
    if dtype != x.dtype:
      x = x.astype(dtype)
    shape = desc[2:]
    if len(shape) != len(x.shape):
      _raise_argument_error(inv,
                            f"rank mismatch {len(x.shape)} vs {len(shape)}")
    for exp_dim, act_dim in zip(shape, x.shape):
      if exp_dim is not None and exp_dim != act_dim:
        _raise_argument_error(inv,
                              f"shape mismatch {x.shape} vs {tuple(shape)}")
  actual_dtype = x.dtype
  for match_dtype, element_type in DTYPE_TO_HAL_ELEMENT_TYPE:
    if match_dtype == actual_dtype:
      break
  else:
    _raise_argument_error(inv, f"unsupported numpy dtype {x.dtype}")
  t.push_buffer_view(inv.device, x, element_type)


PYTHON_TO_VM_CONVERTERS = {
    bool: _bool_to_vm,
    int: _int_to_vm,
    float: _float_to_vm,
    list: _list_to_vm,
    tuple: _tuple_to_vm,
    dict: _dict_to_vm,
    str: _str_to_vm,
    np.ndarray: _ndarray_to_vm,
}

# VM to Python converters. All take:
#   inv: Invocation
#   vm_list: VmVariantList to read from
#   vm_index: Index in the vm_list to extract
#   desc: The ABI descriptor list (or None if in dynamic mode)
# Return the corresponding Python object.


def _vm_to_ndarray(inv: Invocation, vm_list: VmVariantList, vm_index: int,
                   desc):
  return vm_list.get_as_ndarray(vm_index)


VM_TO_PYTHON_CONVERTERS = {
    "ndarray": _vm_to_ndarray,
}

ABI_TYPE_TO_DTYPE = {
    # TODO: Others.
    "f32": np.float32,
    "i32": np.int32,
}

# NOTE: Numpy dtypes are not hashable and exist in a hierarchy that should
# be queried via isinstance checks. This should be done as a fallback but
# this is a linear list for quick access to the most common.
DTYPE_TO_HAL_ELEMENT_TYPE = (
    # TODO: Others.
    (np.float32, HalElementType.FLOAT_32),
    (np.float64, HalElementType.FLOAT_64),
)


def _raise_argument_error(inv: Invocation, summary: str):
  # TODO: Do some fancier back-tracking in reporting error.
  raise ValueError(f"Error passing argument: {summary}")


def _raise_return_error(inv: Invocation, summary: str):
  # TODO: Do some fancier back-tracking in reporting error.
  raise ValueError(f"Error processing function return: {summary}")


def _merge_python_sequence_to_vm(inv: Invocation, vm_list, py_list, descs):
  # For dynamic mode, just assume we have the right arity.
  if descs is None:
    descs = [None] * len(py_list)
  elif len(py_list) != len(descs):
    _raise_argument_error(
        inv, f"mismatched function call arity: "
        f"expected={descs}, got={py_list}")
  for py_value, desc in zip(py_list, descs):
    py_type = py_value.__class__
    try:
      converter = PYTHON_TO_VM_CONVERTERS[py_type]
    except KeyError:
      _raise_argument_error(inv, f"cannot map Python type to VM: {py_type}")
    converter(inv, vm_list, py_value, desc)


def _extract_vm_sequence_to_python(inv: Invocation, vm_list, descs):
  vm_list_arity = len(vm_list)
  if descs is None:
    descs = [None] * vm_list_arity
  elif vm_list_arity != len(descs):
    _raise_return_error(
        inv, f"mismatched return arity: {vm_list_arity} vs {len(descs)}")
  results = []
  for vm_index, desc in zip(range(vm_list_arity), descs):
    if desc is None:
      # Dynamic (non reflection mode).
      _raise_return_error(
          inv, "function has no reflection data, which is not yet supported")
    vm_type = desc[0]
    try:
      converter = VM_TO_PYTHON_CONVERTERS[vm_type]
    except KeyError:
      _raise_return_error(inv, f"cannot map VM type to python: {vm_type}")
    results.append(converter(inv, vm_list, vm_index, desc))
  return results
