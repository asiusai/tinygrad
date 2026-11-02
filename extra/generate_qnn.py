"""Generate the QNN 2.42 API bindings from a local QAIRT SDK."""
import os
from pathlib import Path
from tinygrad.runtime.support.autogen import gen

if __name__ == '__main__':
  headers = Path(os.environ['QNN_SDK_ROOT']) / 'include/QNN'
  print(gen('qnn', [headers / 'QnnInterface.h', headers / 'System/QnnSystemInterface.h', headers / 'HTP/QnnHtpDevice.h'],
            args=[f'-I{headers}', f'-I{headers / "HTP"}', f'-I{headers / "System"}'],
            patterns=['^QnnInterface_t$', '^QnnSystemInterface_t$', '^QnnHtpDevice_Infrastructure_t$'], macros=False), end='')
