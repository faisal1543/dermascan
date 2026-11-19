"""Fixed class order (Ground rule 4). Never reorder: saved models depend on it."""

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MALIGNANT = ["akiec", "bcc", "mel"]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASSES)}
