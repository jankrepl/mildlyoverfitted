import re
import pandas as pd
from sklearn.metrics import classification_report


def check_valid(annots: list[str]) -> bool:
    allowed_pattern = re.compile(r"^(O$|B-.+$|I-.+$)")

    annots = ["O"] + annots
    n = len(annots)

    if any(allowed_pattern.match(annot) is None for annot in annots):
        return False

    for i in range(1, n):
        annot = annots[i]

        if annot.startswith("I-"):
            if annots[i - 1] == "O" or annots[i - 1][2:] != annot[2:]:
                return False


    return True

def get_etypes(annots: list[str]) -> list[None | str]:
    return [annot[2:] if annot != "O" else None for annot in annots]


def get_entities(annots: list[str]) -> list[dict[str, int | str]]:
    if not check_valid(annots):
        raise ValueError("Invalid input.")

    annots = ["O"] + annots + ["O"]
    etypes = get_etypes(annots)
    n = len(annots)

    start_patterns = {
        ("O", "B-"),  # ["O", "B-LOC"]
        ("B-", "B-"),  # ["B-PERSON", "B-LOC"]
        ("I-", "B-"),  # ["B-LOC", "I-LOC", "B-PERSON"]
    }

    end_patterns = {
        ("I-", "O"), # ["B-LOC", "I-LOC", "O"]
        ("B-", "O"), # ["B-LOC", "O"]
        ("B-", "B-"),  # ["B-PERSON", "B-LOC"]
        ("I-", "B-"),  # ["B-LOC", "I-LOC", "B-PERSON"]
    }

    entities: list[dict[str, int | str]] = []


    i = 1
    start = None

    while i < n:
        prev, curr = annots[i - 1], annots[i]
        pattern = (prev[:2], curr[:2])


        if pattern in end_patterns and start is not None:
            entities.append(
                {
                    "start": start - 1,
                    "end": i - 2,
                    "etype": etypes[i - 1],

                }
            )

            start = None

        if pattern in start_patterns:
            start = i

        i += 1

    return entities


def get_report(annots_true: list[str], annots_pred: list[str]) -> dict:
    if len(annots_true) != len(annots_pred):
        raise ValueError("Unequal lengths")

    entities_true = pd.DataFrame(get_entities(annots_true))
    entities_pred = pd.DataFrame(get_entities(annots_pred))


    entities_true = entities_true.rename(columns={"etype": "etype_true"})
    entities_pred = entities_pred.rename(columns={"etype": "etype_pred"})

    df_merge = entities_true.merge(entities_pred, on=["start", "end"], how="outer")
    df = df_merge.fillna("")

    labels = (set(df["etype_true"].tolist()) | set(df["etype_pred"].tolist())) - {""}

    report = classification_report(
        df["etype_true"],
        df["etype_pred"],
        output_dict=True,
        labels=list(labels),
    )
    return report
