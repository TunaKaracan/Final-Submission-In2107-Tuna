import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import supervision as sv
from supervision.metrics import MeanAveragePrecision


def calc_accuracy(gt, pred):
	return accuracy_score(gt, pred)


def calc_f1(gt, pred, pos_label):
	return f1_score(gt, pred, pos_label=pos_label)


def calc_bleu(refs, cands):
	bleu_scores = []

	for (ref, cand) in zip(refs, cands):
		ref_tokens = word_tokenize(ref)
		cand_tokens = word_tokenize(cand)

		weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]

		bleu_scores.append(sentence_bleu([ref_tokens], cand_tokens, weights=weights))

	return bleu_scores


def calc_map(pred_boxes, pred_classes, true_boxes, true_classes):
	preds = _boxes_to_detections(pred_boxes, pred_classes)
	targets = _boxes_to_detections(true_boxes, true_classes)

	metric = MeanAveragePrecision()
	result = metric.update(preds, targets).compute()

	return result


def _boxes_to_detections(boxes, class_ids=None):
	xyxy = np.array(boxes, dtype=np.float32)
	class_id_arr = np.array(class_ids, dtype=int)

	confidence = np.ones(len(boxes), dtype=np.float32)  # fixed confidence = 1.0

	return sv.Detections(
		xyxy=xyxy,
		class_id=class_id_arr,
		confidence=confidence
	)