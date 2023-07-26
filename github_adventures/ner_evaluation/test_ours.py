import pytest
from seqeval.metrics import classification_report as cr
from seqeval.scheme import IOB2
from ours import check_valid, get_entities, get_etypes, get_report


@pytest.mark.parametrize(
    "inp,out",
    [
        ([], True),
        (["NONSENSE", "O"], False),
        (["O", "O", "O"], True),
        (["B-"], False),
        (["O", "I-ORG", "O"], False),
        (["O", "B-ORG", "I-PERSON"], False),
        (["O", "B-ORG", "B-PERSON"], True),
        (["O", "SOMETHING", "B-PERSON"], False),
        (["O-", "O", "O"], False),
        (["B-A", "O", "B-T"], True),
        (["I-a", "B-a", "B-a", "I-a", "I-a", "O"], False),
    ],
)
def test_check_valid(inp, out):
    assert check_valid(inp) == out


@pytest.mark.parametrize(
    "inp,out",
    [
        ([], []),
        (["O", "O", "O"], [None, None, None]),
        (["O", "B-ORG", "O"], [None, "ORG", None]),
        (["O", "B-ORG", "B-ORG"], [None, "ORG", "ORG"]),
        (["O", "B-PERSON", "I-PERSON"], [None, "PERSON", "PERSON"]),
        (["B-A", "O", "B-T"], ["A", None, "T"]),
    ],
)
def test_get_etypes(inp, out):
    assert get_etypes(inp) == out


@pytest.mark.parametrize(
    "inp,out",
    [
        (["O", "O", "O"], []),
        (["O", "B-ORG", "O"], [{"start": 1, "end": 1, "etype": "ORG"}]),
        (
            ["O", "B-ORG", "B-ORG"],
            [
                {"start": 1, "end": 1, "etype": "ORG"},
                {"start": 2, "end": 2, "etype": "ORG"},
            ],
        ),
        (["O", "B-PERSON", "I-PERSON"], [{"start": 1, "end": 2, "etype": "PERSON"}]),
        (
            ["B-A", "O", "B-T"],
            [
                {"start": 0, "end": 0, "etype": "A"},
                {"start": 2, "end": 2, "etype": "T"},
            ],
        ),
        (["B-LOC", "I-LOC", "I-LOC"], [{"start": 0, "end": 2, "etype": "LOC"}]),
        (
            ["B-A", "I-A", "B-T"],
            [
                {"start": 0, "end": 1, "etype": "A"},
                {"start": 2, "end": 2, "etype": "T"},
            ],
        ),
    ],
)
def test_get_entities(inp, out):
    assert get_entities(inp) == out


@pytest.mark.parametrize(
    "annots_true,annots_pred",
    [
        (
            ["O", "B-PERSON", "I-PERSON", "O"],
            ["O", "B-PERSON", "I-PERSON", "O"],
        ),
        (
            ["O", "B-PERSON", "I-PERSON", "B-LOC"],
            ["O", "B-PERSON", "I-PERSON", "O"],
        ),
        (
            ["O", "B-PERSON", "I-PERSON", "O"],
            ["O", "O", "B-PERSON", "O"],
        ),
        (
            ["O", "B-PERSON", "I-PERSON", "O"],
            ["O", "O", "B-PERSON", "O"],
        ),
        (
            ["B-PERSON", "B-LOC", "I-LOC", "B-DATE"],
            ["B-PERSON", "B-DATE", "B-PERSON", "B-DATE"],
        ),
        (
            ["B-PERSON", "I-PERSON", "I-PERSON", "O", "O", "B-LOC", "B-DATE"],
            ["B-PERSON", "I-PERSON", "I-PERSON", "O", "O", "B-LOC", "B-DATE"],
        ),
        (
            ["B-PERSON", "O", "O", "O", "B-LOC", "I-LOC", "O", "B-LOC"],
            ["B-PERSON", "O", "B-DATE", "O", "B-LOC", "I-LOC", "I-LOC", "I-LOC"],
        ),
        (
            ["B-PERSON", "I-PERSON", "O", "B-LOC", "I-LOC", "O", "B-PERSON", "B-PERSON", "B-LOC"],
            ["B-PERSON", "I-PERSON", "O", "B-LOC", "B-LOC", "O", "B-PERSON", "B-PERSON", "B-LOC"],
        ),
    ]
)
def test_get_report(annots_true, annots_pred):
    report = get_report(annots_true, annots_pred)
    seqeval_report = cr([annots_true], [annots_pred], scheme=IOB2, mode="strict", output_dict=True)

    keys_to_delete = {"accuracy", "micro avg"}

    for rep in (report, seqeval_report):
        for key in keys_to_delete:
            try:
                rep.pop(key)
            except KeyError:
                pass


    assert report == seqeval_report
