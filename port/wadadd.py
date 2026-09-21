from __future__ import annotations

import struct
import sys
from typing import cast


def main() -> None:
    src, lump, name, out = sys.argv[1:5]
    if len(name) > 8:
        raise SystemExit(f"lump name too long: {name}")
    wad = open(src, "rb").read()
    data = open(lump, "rb").read()
    ident, nlumps, dirofs = cast(tuple[bytes, int, int],
                                 struct.unpack_from("<4sII", wad, 0))
    if ident not in (b"IWAD", b"PWAD"):
        raise SystemExit(f"{src}: not a WAD file")
    if dirofs + nlumps * 16 != len(wad):
        raise SystemExit(f"{src}: lump directory is not at end of file")
    body = bytearray(wad[:dirofs]) + data
    struct.pack_into("<4sII", body, 0, ident, nlumps + 1, len(body))
    directory = wad[dirofs:] + struct.pack(
        "<II8s", dirofs, len(data), name.upper().encode().ljust(8, b"\0")
    )
    with open(out, "wb") as f:
        f.write(body)
        f.write(directory)
    print(
        f"{out}: {nlumps} -> {nlumps + 1} lumps, "
        f"added {name.upper()} ({len(data)} bytes @ {dirofs:#x})"
    )


if __name__ == "__main__":
    main()
