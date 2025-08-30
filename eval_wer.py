import json
from jiwer import wer, RemovePunctuation, ToLowerCase, Strip, RemoveMultipleSpaces, Compose

tr = Compose([RemoveMultipleSpaces(), Strip(), ToLowerCase(), RemovePunctuation()])

refs = json.load(open("refs.json"))
hyps = json.load(open("hyps.json"))

ref_texts = [tr(refs[k]) for k in refs]
hyp_texts = [tr(hyps.get(k,"")) for k in refs]

print("N =", len(ref_texts))
print("WER:", wer(ref_texts, hyp_texts))