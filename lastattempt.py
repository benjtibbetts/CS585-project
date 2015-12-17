features = []

def make_featlist(sent1, sent2, sent1_tag, sent2_tag, label):
	feat = {}
	if label in para_label:
		feat["label"] = 1
	if label in nonpara_label:
		feat["label"] = 0

	sent1_list = tokenize(sent1)
	sent2_list = tokenize(sent2)
	sent1_pos = getPOS(tokenize(sent1_tag))
	sent2_pos = getPOS(tokenize(sent2_tag))

	feat["sent_cos"] = get_cosine(sent1_list, sent2_list)
	feat["pos_cos"] = get_cosine(sent1_pos, sent2_pos)
	features.append(feat)

def getPOS(sent_tag):
	pos = []
	for t in sent_tag:
		tag_info = t.split("/")
		pos.append(tag_info[2])
	return pos

def get_cosine(sent1, sent2):
	vec1 = Counter(sent1)
	vec2 = Counter(sent2)

	intersection = set(vec1.keys()) & set(vec2.keys())
	numerator = sum([vec1[x] * vec2[x] for x in intersection])

	sum1 = sum([vec1[x]**2 for x in vec1.keys()])
	sum2 = sum([vec2[x]**2 for x in vec2.keys()])
	denominator = math.sqrt(sum1) * math.sqrt(sum2)

	if not denominator:
		return 0.0
	else:
		return float(numerator) / denominator

def tokenize(doc):
    tokens = doc.split()
    lowered_tokens = [t.lower() for t in tokens]
    return lowered_tokens