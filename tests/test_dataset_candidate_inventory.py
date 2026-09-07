from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs" / "discovered-dataset-candidates.md"

EXPECTED_CANDIDATES = (
    "YCB-Sight",
    "PHAC-2",
    "TacQuad",
    "Touch in the Wild",
    "VisTouch",
    "Eagle Shoal Visual-Tactile Dataset",
    "Tactile MNIST",
    "Feeling of Success",
    "ObjectFolder 2.0",
    "ObjectFolder Real",
    "Touch and Go",
    "YCB-Slide",
    "SSVTP",
    "VisGel",
    "PhysiCLeAR / Octopi",
)

REQUIRED_EVIDENCE_FIELDS = (
    "Primary sources and snapshot",
    "Contents and schema",
    "Open-access and license evidence",
    "Popularity and adoption",
    "Relevance and overlap",
    "Access cost and constraints",
    "Likely integration surface and effort",
    "Step 10 questions",
)


def _candidate_sections(text: str) -> dict[str, str]:
    marker = "### Candidate: "
    sections: dict[str, str] = {}
    for raw_section in text.split(marker)[1:]:
        title, _, body = raw_section.partition("\n")
        sections[title.strip()] = body
    return sections


def test_inventory_has_one_complete_record_per_candidate() -> None:
    sections = _candidate_sections(INVENTORY.read_text(encoding="utf-8"))

    assert tuple(sections) == EXPECTED_CANDIDATES
    for candidate, section in sections.items():
        for field in REQUIRED_EVIDENCE_FIELDS:
            assert f"- **{field}:**" in section, f"{candidate} is missing {field}"


def test_inventory_is_explicitly_research_not_support() -> None:
    text = INVENTORY.read_text(encoding="utf-8")

    assert "it is research input, not a claim that the toolkit supports" in text
    assert "no candidate files were downloaded or added to the package" in text
    assert "Step 10 should" in text


def test_inventory_is_linked_without_changing_published_dataset_entries() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    coverage = (ROOT / "docs" / "dataset-coverage.md").read_text(encoding="utf-8")

    inventory_link = (
        "[discovered dataset candidate inventory](docs/discovered-dataset-candidates.md)"
    )
    assert inventory_link in readme
    assert "[candidate inventory](discovered-dataset-candidates.md)" in coverage
    assert "### Candidate:" not in readme
