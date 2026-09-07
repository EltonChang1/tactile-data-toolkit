import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs" / "discovered-dataset-candidates.md"
SHORTLIST = ROOT / "docs" / "discovered-dataset-shortlist.md"

EXPECTED_DECISIONS = {
    "YCB-Sight": "S12",
    "PHAC-2": "S11",
    "TacQuad": "S12",
    "Touch in the Wild": "DEFER",
    "VisTouch": "DEFER",
    "Eagle Shoal Visual-Tactile Dataset": "DEFER",
    "Tactile MNIST": "S11",
    "Feeling of Success": "DEFER",
    "ObjectFolder 2.0": "DEFER",
    "ObjectFolder Real": "REJECT",
    "Touch and Go": "REJECT",
    "YCB-Slide": "REJECT",
    "SSVTP": "REJECT",
    "VisGel": "REJECT",
    "PhysiCLeAR / Octopi": "REJECT",
}

EXPECTED_PACKETS = (
    "S11.1 — Tactile MNIST real single-touch 64×64",
    "S11.2 — PHAC-2 version 1.0",
    "S12.1 — YCB-Sight real and simulated releases",
    "S12.2 — TacQuad",
)

PACKET_FIELDS = (
    "Scope and evidence",
    "Files and API",
    "Acquisition and integrity",
    "Normalized mapping",
    "Tests and examples",
    "License/provenance behavior",
    "Acceptance criteria",
)


def _inventory_names(text: str) -> set[str]:
    return set(re.findall(r"^### Candidate: (.+)$", text, flags=re.MULTILINE))


def _decision_rows(text: str) -> dict[str, str]:
    decisions: dict[str, str] = {}
    pattern = re.compile(r"^\| (?P<name>[^|]+?) \|.*\| `(?P<outcome>S11|S12|DEFER|REJECT)` \|.*\|$")
    for line in text.splitlines():
        match = pattern.fullmatch(line)
        if match:
            decisions[match.group("name").strip()] = match.group("outcome")
    return decisions


def _packet_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    for raw_section in text.split("### ")[1:]:
        title, _, body = raw_section.partition("\n")
        if title.startswith(("S11.", "S12.")):
            sections[title.strip()] = body
    return sections


def test_every_inventory_candidate_has_one_vetting_decision() -> None:
    inventory = INVENTORY.read_text(encoding="utf-8")
    shortlist = SHORTLIST.read_text(encoding="utf-8")

    assert _decision_rows(shortlist) == EXPECTED_DECISIONS
    assert set(EXPECTED_DECISIONS) == _inventory_names(inventory)


def test_shortlist_is_bounded_and_prioritized() -> None:
    decisions = _decision_rows(SHORTLIST.read_text(encoding="utf-8"))

    assert [name for name, outcome in decisions.items() if outcome == "S11"] == [
        "PHAC-2",
        "Tactile MNIST",
    ]
    assert [name for name, outcome in decisions.items() if outcome == "S12"] == [
        "YCB-Sight",
        "TacQuad",
    ]
    assert sum(outcome == "DEFER" for outcome in decisions.values()) == 5
    assert sum(outcome == "REJECT" for outcome in decisions.values()) == 6


def test_each_selected_release_has_a_complete_implementation_packet() -> None:
    sections = _packet_sections(SHORTLIST.read_text(encoding="utf-8"))

    assert tuple(sections) == EXPECTED_PACKETS
    for packet, section in sections.items():
        for field in PACKET_FIELDS:
            assert f"- **{field}:**" in section, f"{packet} is missing {field}"


def test_decisions_are_linked_and_do_not_claim_runtime_support() -> None:
    inventory = INVENTORY.read_text(encoding="utf-8")
    shortlist = SHORTLIST.read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    coverage = (ROOT / "docs" / "dataset-coverage.md").read_text(encoding="utf-8")

    assert "[Step 10 qualification record](discovered-dataset-shortlist.md)" in inventory
    assert "this Step 10 decision does not raise" in shortlist
    assert "[Step 10 qualification record](docs/discovered-dataset-shortlist.md)" in readme
    assert "[qualification record](discovered-dataset-shortlist.md)" in coverage
