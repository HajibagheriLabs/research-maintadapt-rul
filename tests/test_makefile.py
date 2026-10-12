import re
from pathlib import Path

MAKEFILE = Path(__file__).resolve().parents[1] / "Makefile"


def test_help_lists_every_target_once() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    targets = re.findall(r"^([a-z][\w-]*):(?!=)", text, flags=re.MULTILINE)
    block = text.split("define HELP\n", 1)[1].split("\nendef", 1)[0]
    listed = re.findall(r"^  ([a-z][\w-]*) ", block, flags=re.MULTILINE)
    assert sorted(listed) == sorted(targets)
    phony = re.search(r"^\.PHONY:(.*)$", text, flags=re.MULTILINE)
    assert phony is not None and sorted(phony.group(1).split()) == sorted(targets)
