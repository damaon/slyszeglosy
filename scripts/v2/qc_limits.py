"""QC limits for the strict annotation pass, from the loose pass's train quantiles (the V1 rule):
token probability and vowel voicing at the 1st percentile, weak-phone fraction at the 99th,
rounded outwards to 0.05. Prints the annotate flags.

Vowel voicing never goes below the V1 limit 0.45: on MC Speech the tracker (pyin, 1024-sample
window) loses voicing on whole utterances of this low (~105 Hz) voice, so the 1st percentile is 0
and would admit clips whose vowels the vocoder would learn as noise.

    python scripts/v2/qc_limits.py ~/sg-data/v2/annotations/summary.json
"""

import json
import math
import sys

q = json.load(open(sys.argv[1]))["splits"]["train"]["qc_quantiles"]
down = lambda x: math.floor(x * 20) / 20
up = lambda x: math.ceil(x * 20) / 20
limits = {"--min-token-probability": down(q["token_probability_mean"]["1"]),
          "--max-weak-phone-fraction": up(q["weak_phone_fraction"]["99"]),
          "--min-vowel-voiced": max(down(q["vowel_voiced"]["1"]), 0.45)}
print(" ".join(f"{k} {v:.2f}" for k, v in limits.items()))
