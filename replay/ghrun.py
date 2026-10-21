"""
Run a replay module and, if it crashes, post the traceback as a GitHub
Actions error annotation (job logs are not readable from the Claude session,
annotations are).

    python -m replay.ghrun replay.agent --mode copy ...
"""

from __future__ import annotations

import runpy
import sys
import traceback


def _escape(s: str) -> str:
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main() -> int:
    module, args = sys.argv[1], sys.argv[2:]
    sys.argv = [module] + args
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return 0
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if code:
            print(f"::error title={module} exited {code}::{_escape(str(e.code))}", flush=True)
        return code
    except Exception:
        tb = traceback.format_exc().strip().splitlines()
        print(f"::error title={module} {' '.join(args)}::{_escape(chr(10).join(tb[-25:]))}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
