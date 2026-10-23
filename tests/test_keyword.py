import pytest

from kneemri.labels.keyword import label_report, to_soft

CASES = [
    ("Complete ACL tear. Small joint effusion. No fracture. Medial meniscus intact, lateral meniscus tear.",
     {"ACL": 1, "Effusion": 1, "Fracture": 0, "Medial Meniscus": 0, "Lateral Meniscus": 1}),
    ("Kein Hinweis auf eine Fraktur oder ein Bone-bruise. Stigmata der medial betonten Gonarthrose. Kleiner Gelenkerguss.",
     {"Fracture": 0, "Contusion": 0, "Medial OA": 1, "Effusion": 1}),
    ("SAĞ DİZ MRG. Medial menisküs posterior hornunda horizontal yırtık mevcuttur. Kırık izlenmedi. Baker kisti.",
     {"Medial Meniscus": 1, "Fracture": 0, "Baker's": 1}),
    ("Hondromalacija III° trohleje femura te obje fasete patele. Bez izljeva.", {"PF OA": 1, "Effusion": 0}),
    ("Rotura completa del ligamento cruzado anterior. Edema óseo en cóndilo femoral lateral. Derrame articular.",
     {"ACL": 1, "Contusion": 1, "Effusion": 1}),
    ("No fracture, contusion or effusion.", {"Fracture": 0, "Contusion": 0, "Effusion": 0}),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_multilingual_reports(text, expected):
    got = label_report(text)
    assert {k: got[k] for k in expected} == expected


def test_unmentioned_findings_are_none():
    assert label_report("Small joint effusion.")["ACL"] is None


def test_soft_targets():
    assert (to_soft(1), to_soft(0), to_soft(None)) == (0.95, 0.02, 0.05)
