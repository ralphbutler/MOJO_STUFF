from std.ffi import external_call
from std.subprocess import run


def main() raises:
    print("hello from mojo harness")
    var pid = external_call["getpid", Int32]()
    print("pid:", pid)
    var out: String = run("echo subprocess-ok")
    print("subprocess:", out)
