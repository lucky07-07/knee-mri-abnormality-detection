"""Shared constants: the 12 findings and the 6 series slots of a knee MRI study."""

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

# One series per anatomical plane x fluid sensitivity. Missing slots are zero-filled and masked in the model.
SLOTS = ["sag_fluid", "sag_nonfluid", "cor_fluid", "cor_nonfluid", "ax_fluid", "ax_nonfluid"]

# Short keys used in the LLM grading prompt (same order as LABELS).
LLM_KEYS = ["ACL", "MCL", "MM", "LM", "MOA", "LOA", "PFOA", "EFF", "SYN", "BAK", "CON", "FX"]
