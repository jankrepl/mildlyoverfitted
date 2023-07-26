import pprint
import evaluate


metric = evaluate.load("seqeval")


# Tom Cruise is great
annots_true = ["B-PERSON", "I-PERSON", "O", "O"]
# annots_pred = ["B-PERSON", "I-PERSON", "O", "O"]
# annots_pred = ["O", "O", "O", "O"]
# annots_pred = ["B-PERSON", "O", "O", "O"]
annots_pred = ["B-LOCATION", "I-LOCATION", "O", "O"]


result = metric.compute(references=[annots_true], predictions=[annots_pred])

pprint.pprint(result)
