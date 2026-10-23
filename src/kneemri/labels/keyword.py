"""Multilingual rule-based labeler for knee MRI reports.

Text is accent-stripped and casefolded, then split into sentences and clauses. Each of the 12 findings is
marked 1 (asserted), 0 (negated) or None (not mentioned). This is the fallback label source: the LLM and
teacher sources are blended on top of it (see ``kneemri.labels.blend``).
"""
import re
import unicodedata

from kneemri.constants import LABELS


def normalize(text):
    t = unicodedata.normalize("NFKD", str(text))
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return t.casefold().replace("ı", "i")


def _rx(*parts):
    return re.compile("|".join(parts))


ACL = _rx(r"\bacl\b", r"\blca\b", r"\bvkb\b", r"anterior cruciate", r"vordere[snm]? kreuzband", r"kreuzband vorn",
          r"on capraz", r"cruzado anterior", r"crociato anteriore", r"croise anterieur", r"prednj\w* (ukrizen|krizn)",
          r"προσθιος χιαστος", r"предн\w* кръстн", r"voorste kruisband", r"前交叉韧带")
MCL = _rx(r"\bmcl\b", r"medial collateral", r"tibial collateral", r"innenband", r"\bmkb\b", r"medial kollateral",
          r"ic yan bag", r"colateral medial", r"collaterale mediale", r"collateral (medial|interne)", r"\blcm\b",
          r"medijaln\w* kolateral", r"unutarnj\w* kolateral", r"εσω πλαγιος", r"μεσ\w* πλαγιος",
          r"медиалн\w* колатерал", r"mediale collateral", r"内侧副韧带")
INJURY = _rx(r"tear", r"torn", r"ruptur", r"disrupt", r"sprain", r"partial", r"insufficien", r"discontinu",
             r"\blesion", r"lasion", r"\briss", r"einriss", r"yirti", r"kopma", r"rotura", r"desgarr", r"lesao",
             r"lesione", r"dechirure", r"puknu", r"lezij", r"ρηξ", r"разкъсв", r"руптур", r"scheur", r"distorsion",
             r"zerrung", r"esguince", r"distorsione", r"撕裂", r"断裂", r"损伤")
MENISCUS = _rx(r"menisc", r"menisk", r"menisq", r"μηνισκ", r"мениск", r"半月板")
MEN_TEAR = _rx(INJURY.pattern, r"grade (3|iii)\b", r"grad (3|iii)\b", r"derece 3", r"grado (3|iii)\b")
MEDIAL = _rx(r"medial", r"\binnen", r"medijal", r"\binterno", r"\binterne", r"\bic\b", r"μεσ", r"εσω", r"медиал",
             r"内侧")
LATERAL = _rx(r"lateral", r"\baussen", r"\bexterno", r"\bexterne", r"\bdis\b", r"εξω", r"латерал", r"外侧")
BOTH = _rx(r"\bboth\b", r"\bbeide", r"\bbds\b", r"beidseit", r"her iki", r"\bambos\b", r"\bambas\b", r"\boba\b",
           r"\bobje\b", r"entrambi", r"\bdeux\b", r"και των δυο", r"двата", r"二者", r"双侧")
OA = _rx(r"arthros", r"artros", r"osteoarthr", r"osteoartr", r"gonarthr", r"gonartr", r"chondropath",
         r"chondromalaz", r"chondromalac", r"hondromalac", r"condromala", r"kondromala", r"kondropat",
         r"chondrosis", r"cartilage (loss|thinning|defect)", r"knorpel(schaden|verlust|ausdunnung)", r"kikirdak",
         r"cartilag", r"osteophyt", r"osteofit", r"οστεοφυτ", r"остеофит", r"αρθρωσ", r"артроз", r"хондр",
         r"χονδρ", r"骨关节炎", r"退行性", r"软骨")
PF = _rx(r"patell", r"patel", r"femoropatell", r"patellofemor", r"patelofemor", r"trochle", r"retropatell",
         r"επιγονατ", r"пател", r"髌")
TIBIOFEM = _rx(r"femorotib", r"tibiofem", r"femoro-tib", r"tibio-fem", r"compartment", r"kompartiment",
               r"kompartman", r"compartimento", r"compartiment", r"condyl", r"kondyl", r"kondil", r"plateau",
               r"platos", r"plato")
EFFUSION = _rx(r"effusion", r"erguss", r"efuzyon", r"eklem sivi", r"sivi artis", r"derrame", r"versamento",
               r"izljev", r"συλλογ", r"излив", r"hydrops", r"vocht", r"epanchement", r"积液")
SYNOVITIS = _rx(r"synovit", r"sinovit", r"synovialit", r"συνοβιτ", r"синовит", r"滑膜炎",
                r"synovial (thicken|hypertroph|prolif)", r"sinovyal kalinlas")
BAKER = _rx(r"baker", r"popliteal\w* (cyst|zyst|kist)", r"poplitealzyst", r"quiste popliteo", r"cisto popliteo",
            r"cisti poplitea", r"kyste poplite", r"poplitealn\w* cist", r"бейкер", r"腘窝囊肿", r"贝克")
CONTUSION = _rx(r"contusion", r"bone[ -]bruise", r"bruise", r"marrow (o)?edema", r"knochenmark\w*odem",
                r"knochenmarksodem", r"kontusion", r"kontuzyon", r"kemik iligi odem", r"edema oseo",
                r"edema medular", r"edema da medula", r"edema osseo", r"contusione", r"kontuzij",
                r"edem kostane srzi", r"οστικ\w* οιδημα", r"οιδημα του μυελου", r"контузи", r"костно\w* оток",
                r"beenmergoedeem", r"oedeme osseux", r"骨挫伤", r"骨髓水肿")
FRACTURE = _rx(r"fractur", r"fraktur", r"kirik", r"fratura", r"frattura", r"prijelom", r"καταγμ", r"фрактур",
               r"счупв", r"breuk", r"avulsion", r"骨折")

NEG = _rx(r"\bno\b", r"\bnot\b", r"without", r"negative for", r"free of", r"\bintact", r"unremarkable", r"absent",
          r"\bnormal", r"\bkein", r"\bohne\b", r"\bnicht\b", r"unauffallig", r"intakt", r"regelrecht", r"\byok",
          r"izlenmedi", r"izlenmemis", r"saptanmad", r"saptanmam", r"gorulmedi", r"rastlanmad", r"degil",
          r"\bolagan", r"saglam", r"\bsin\b", r"ausencia", r"descart", r"integr", r"conservad", r"preservad",
          r"\bsem\b", r"\bnao\b", r"\bnon\b", r"assenza", r"nessun", r"regolar", r"pas de", r"\bsans\b",
          r"absence", r"\bnema\b", r"\bbez\b", r"\bnije\b", r"uredn", r"ocuvan", r"\bδεν\b", r"χωρις",
          r"ουδεμια", r"φυσιολογ", r"ακεραι", r"\bбез\b", r"\bняма\b", r"\bне\b", r"запазен", r"нормал",
          r"\bgeen\b", r"zonder", r"normaal", r"无", r"未见", r"正常", r"未发现")
# a sentence that STARTS with one of these negates its whole list ("No fracture, contusion or effusion.")
LEAD_NEG = re.compile(r"^\W*(no|without|kein\w*|ohne|sin|sem|nema|bez|без|няма|geen|pas de|nessun\w*|δεν|χωρις|无|未见)\b")
SENT_SPLIT = re.compile(r"[.;\n!?]+|\s-\s|\s(?:but|however|jedoch|aber|ancak|fakat|pero|mas|ali|αλλα|но|maar|mais|ma)\s")


def _clauses(text):
    for sent in SENT_SPLIT.split(normalize(text)):
        sent = sent.strip()
        if not sent:
            continue
        lead_neg = bool(LEAD_NEG.search(sent))
        for clause in sent.split(","):
            clause = clause.strip()
            if clause:
                yield clause, lead_neg or bool(NEG.search(clause))


def label_report(text):
    # Returns {label: 1 | 0 | None}. 1 wins over 0 if a finding is both asserted and negated somewhere.
    out = {k: None for k in LABELS}

    def mark(k, neg):
        v = 0 if neg else 1
        out[k] = v if out[k] is None else max(out[k], v)

    for c, neg in _clauses(text):
        if ACL.search(c) and (INJURY.search(c) or neg):
            mark("ACL", neg)
        if MCL.search(c) and (INJURY.search(c) or neg):
            mark("MCL", neg)
        if MENISCUS.search(c) and (MEN_TEAR.search(c) or neg):
            both = BOTH.search(c)
            if MEDIAL.search(c) or both:
                mark("Medial Meniscus", neg)
            if LATERAL.search(c) or both:
                mark("Lateral Meniscus", neg)
        if OA.search(c):
            pf, tf = PF.search(c), TIBIOFEM.search(c)
            if pf:
                mark("PF OA", neg)
            if tf or not pf:
                both = BOTH.search(c)
                if MEDIAL.search(c) or both:
                    mark("Medial OA", neg)
                if LATERAL.search(c) or both:
                    mark("Lateral OA", neg)
        for k, rx in (("Effusion", EFFUSION), ("Synovitis", SYNOVITIS), ("Baker's", BAKER),
                      ("Contusion", CONTUSION), ("Fracture", FRACTURE)):
            if rx.search(c):
                mark(k, neg)
    return out


def to_soft(value, pos=0.95, neg=0.02, unmentioned=0.05):
    """Maps a hard keyword label (1 / 0 / None) to a soft training target."""
    if value is None:
        return unmentioned
    return pos if value == 1 else neg
