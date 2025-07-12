import os
import re
import csv
import json
import textwrap
from time import sleep

import numpy as np
import cv2
from openai import OpenAI

import metrics

disease2id = {
	"Aortic enlargement": 0,
	"Cardiomegaly": 1,
	"Pulmonary fibrosis": 2,
	"Mass": 3,
	"Pleural effusion": 4,
	"Rib fracture": 5,
	"Other lesion": 6,
	"Infiltration": 7,
	"Lung Opacity": 8,
	"Consolidation": 9,
	"Calcification": 10,
	"ILD": 11,
	"Pleural thickening": 12,
	"Pneumonia": 13,
	"Emphysema": 14,
	"Pneumothorax": 15,
	"Lung cyst": 16
}


def create_user_prompt(image):
	user_prompt = [{
		'type': 'input_image',
		'image_url': f'data:image/png;base64,{image}',
		'detail': 'high'
	}]

	return user_prompt


def create_preds_csv(gpt_outputs, names_and_images, experiment):
	results = []

	if experiment == 'chest_xrays':
		for idx, gpt_output in enumerate(gpt_outputs):
			lines = re.split(r'\n\s*\n', gpt_output.strip())[0].strip().splitlines()

			diagnosis = ''
			bboxes = ''
			for line in lines:
				if line.startswith('Diagnosis:'):
					match = re.search(r'Diagnosis:\s*(\w+)', line)
					if match:
						diagnosis = match.group(1)
				elif line.startswith('Bounding boxes:'):
					match = re.search(r'Bounding boxes:\s*(None|(\[.*))', line)
					if match:
						bboxes = match.group(1)

			image_id = list(names_and_images.keys())[idx]
			results.append([image_id, diagnosis, bboxes])

		with open(os.path.join('..', 'outputs', experiment, 'preds.csv'), 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['Image ID', 'Diagnosis', 'Bounding Boxes'])
			writer.writerows(results)
	elif experiment == 'nova_brain':
		for idx, gpt_output in enumerate(gpt_outputs):
			# Split lines by 2 \n's because GPT puts 2 linebreaks for some reason.
			lines = re.split(r'\n\n\s*\n', gpt_output.strip())[0].strip().splitlines()

			description = ''
			bboxes = ''

			for line in lines:
				if line.startswith('Description:'):
					match = re.search(r'Description:\s*(.*)', line)
					if match:
						description = match.group(1)
				elif line.startswith('Bounding boxes:'):
					match = re.search(r'Bounding boxes:\s*(None|(\[.*))', line)
					if match:
						bboxes = match.group(1)

			image_id = list(names_and_images.keys())[idx]
			results.append([image_id, description, bboxes])

		with open(os.path.join('..', 'outputs', experiment, 'preds.csv'), 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['Image ID', 'Description', 'Bounding Boxes'])
			writer.writerows(results)


def create_metrics_csv(experiment_metrics, experiment):
	with open(os.path.join('..', 'outputs', experiment, 'metrics.csv'), 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		if experiment == 'chest_xrays':
			writer.writerow(['Accuracy', 'F1-Score Healthy', 'F1-Score Unhealthy', 'mAP@50:95', 'mAP@50', 'mAP@75'])
			writer.writerow(experiment_metrics)
		elif experiment == 'nova_brain':
			writer.writerow(['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'LLM Score', 'mAP@50:95', 'mAP@50', 'mAP@75'])
			writer.writerow(experiment_metrics)


def get_annotations_and_preds(annotations_path, preds_path, experiment):
	with open(annotations_path, 'r') as annotations_file:
		data = json.load(annotations_file)

	annotations, preds = None, None

	if experiment == 'chest_xrays':
		# Create the annotation dicts and make them appear in alphabetic order
		annotations_bbox2d = dict(sorted({img_id: img_details['bbox_2d'] for img_id, img_details in data.items()}.items()))
		annotations_status = dict(sorted({img_id: img_details['status'] for img_id, img_details in data.items()}.items()))

		preds_bbox2d = {}
		preds_status = {}

		# No need for pandas since csv is small, and we only need to read it
		with open(preds_path, newline='') as preds_file:
			reader = csv.DictReader(preds_file)
			for row in reader:
				preds_bbox2d[row['Image ID']] = row['Bounding Boxes']
				preds_status[row['Image ID']] = row['Diagnosis']

		# Make preds and annotations share the same order
		preds_bbox2d = dict(sorted(preds_bbox2d.items()))
		preds_status = dict(sorted(preds_status.items()))

		# Convert bbox to how annotations is structured
		preds_bbox2d = convert_preds_bbox2d(preds_bbox2d, experiment)

		annotations, preds = (annotations_bbox2d, annotations_status), (preds_bbox2d, preds_status)
	elif experiment == 'nova_brain':
		slices = [list(case['image_findings'].keys()) for case in data.values()]

		# Omit the .png extensions
		annotations_bbox2d = dict(sorted({f'{i[:-4]}': img_details['image_findings'][i]['bbox_2d_gold'] for img_id, img_details in data.items() for i in img_details['image_findings'].keys()}.items()))
		annotations_desc = dict(sorted({f'{i[:-4]}': img_details['image_findings'][i]['caption'] for img_id, img_details in data.items() for i in img_details['image_findings'].keys()}.items()))

		preds_bbox2d = {}
		preds_desc = {}

		with open(preds_path, newline='') as preds_file:
			reader = csv.DictReader(preds_file)
			for row in reader:
				preds_bbox2d[row['Image ID']] = row['Bounding Boxes']
				preds_desc[row['Image ID']] = row['Description']

		# Make preds and annotations share the same order
		preds_bbox2d = dict(sorted(preds_bbox2d.items()))
		preds_desc = dict(sorted(preds_desc.items()))

		# Convert bbox to how annotations is structured
		preds_bbox2d = convert_preds_bbox2d(preds_bbox2d, experiment)

		annotations, preds = (annotations_bbox2d, annotations_desc), (preds_bbox2d, preds_desc)

	return annotations, preds


def convert_preds_bbox2d(preds_bbox2d, experiment):
	new_preds_bbox2d = {}
	if experiment == 'chest_xrays':
		for (img_id, bbox) in preds_bbox2d.items():
			# Match each of the inner lists
			matches = re.findall(r'\[([^\[\]]+?)\]', bbox)
			result = []
			for match in matches:
				parts = [p.strip() for p in match.split(',')]
				nums = parts[:4]
				label = ' '.join(parts[4:])
				result.append([*map(int, nums), label])

			new_preds_bbox2d[img_id] = result
	elif experiment == 'nova_brain':
		for (img_id, bbox) in preds_bbox2d.items():
			# Match each of the inner lists
			matches = re.findall(r'\[([^\[\]]+?)\]', bbox)
			result = []
			for match in matches:
				parts = [p.strip() for p in match.split(',')]
				result.append([*map(int, parts)])
			new_preds_bbox2d[img_id] = result

	return new_preds_bbox2d


def extract_labels_and_coordinates(bbox2d):
	labels = []
	bboxes = []
	for _bbox in bbox2d.values():
		label = []
		bbox = []
		for __bbox in _bbox:
			label.append(disease2id[__bbox[-1]])
			bbox.append(__bbox[:4])
		labels.append(label)
		bboxes.append(bbox)

	return labels, bboxes


def draw_bboxes(annotations_bbox2d, preds_bbox2d, experiment):
	for img_id in annotations_bbox2d:
		img_path = os.path.join('..', 'datasets', experiment, 'images', f'{img_id}.png')

		img = cv2.imread(img_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		# Draw ground truth in green
		annotations_bboxes = annotations_bbox2d[img_id]
		for annotations_bbox in annotations_bboxes:
			if experiment == 'chest_xrays':
				x1, y1, x2, y2, disease = annotations_bbox
			else:
				x1, y1, x2, y2 = annotations_bbox
			x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
			cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
			if experiment == 'chest_xrays':
				cv2.putText(img, disease, (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

		# Draw the predictions in red
		preds_bboxes = preds_bbox2d[img_id]
		for preds_bbox in preds_bboxes:
			if experiment == 'chest_xrays':
				x1, y1, x2, y2, disease = preds_bbox
			else:
				x1, y1, x2, y2 = preds_bbox
			x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
			cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
			if experiment == 'chest_xrays':
				cv2.putText(img, disease, (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)

		out_path = os.path.join('..', 'outputs', experiment, 'images', f'{img_id}.png')
		cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def evaluate_with_llm(annotations_desc, preds_desc):
	SYSTEM_PROMPT = textwrap.dedent("""
		You will be given 2 sentences, 1 ground truth and 1 prediction describing an MRI brain slice.
		You are tasked with determining whether the prediction captures the ground truth and to output a score between [0, 1] and give an explanation regarding your evaluation.
		A score of 0 means the ground truth and the prediction completely contradict each other and a score of 1 means they align perfectly. It should be noted that 1 does not mean prediction matches the ground truth word by word but it matches the ground truth in essence.
		The output format should be plain text, no bold, italic texts or emojis.
		An example is as follows:
		
		Ground Truth: There is a small area of diffusion restriction in the left frontal lobe suggestive of acute infarct.
		Prediction: The MRI shows a subtle abnormality in the left frontal region, which may indicate a possible ischemic event, though findings are inconclusive.
		
		Score: 0.5
		Explanation: The prediction partially captures the ground truth by identifying an abnormality in the same location and suggesting a possible ischemic event, but it is vague and lacks the specificity and certainty present in the ground truth.
		""")

	API_KEY_PATH = os.path.join('..', '..', '..', 'API_KEY.txt')
	with open(API_KEY_PATH, 'r') as api_key_file:
		client = OpenAI(api_key=api_key_file.readline().strip())

	gpt_outputs = []

	for image_id in annotations_desc:
		annotation_desc = annotations_desc[image_id]
		pred_desc = preds_desc[image_id]

		user_prompt = textwrap.dedent(f"""
		Ground Truth: {annotation_desc}
		Prediction: {pred_desc}
		""")

		response = client.responses.create(
			model='gpt-4o-mini',
			store=True,
			input=[
				{'role': 'system', 'content': SYSTEM_PROMPT},
				{'role': 'user', 'content': user_prompt}
			]
		)

		gpt_output = response.output[0].content[0].text
		gpt_outputs.append(gpt_output)

		if image_id != list(annotations_desc.keys())[-1]:
			sleep(10)

	avg_llm = process_llm_evaluation(annotations_desc, preds_desc, gpt_outputs)

	return avg_llm


def process_llm_evaluation(annotations_desc, preds_desc, gpt_outputs):
	results = []
	scores = []

	for idx, image_id in enumerate(annotations_desc):
		lines = re.split(r'\n\s*\n', gpt_outputs[idx].strip())[0].strip().splitlines()

		score = ''
		explanation = ''

		for line in lines:
			if line.startswith('Score:'):
				match = re.search(r'Score:\s*(.*)', line)
				if match:
					score = float(match.group(1))
			elif line.startswith('Explanation:'):
				match = re.search(r'Explanation:\s*(.*)', line)
				if match:
					explanation = match.group(1)

		scores.append(score)
		results.append([image_id, annotations_desc[image_id], preds_desc[image_id], score, explanation])

		with open(os.path.join('..', 'outputs', 'nova_brain', 'llm_evaluation.csv'), 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['Image ID', 'Ground Truth', 'Prediction', 'Score', 'Explanation'])
			writer.writerows(results)

	return np.mean(scores)


def calc_metrics(annotations, preds, experiment):
	experiment_metrics = []

	if experiment == 'chest_xrays':
		annotations_bbox2d, annotations_status = annotations
		preds_bbox2d, preds_status = preds

		# Format all 'Healthy/Unhealthy' to lowercase
		annotations_status_list = [value.lower() for value in annotations_status.values()]
		preds_status_list = [value.lower() for value in preds_status.values()]

		accuracy = metrics.calc_accuracy(annotations_status_list, preds_status_list)
		f1_score_healthy = metrics.calc_f1(annotations_status_list, preds_status_list, pos_label='healthy')
		f1_score_unhealthy = metrics.calc_f1(annotations_status_list, preds_status_list, pos_label='unhealthy')

		annotations_labels, annotations_bboxes = extract_labels_and_coordinates(annotations_bbox2d)
		pred_labels, pred_bboxes = extract_labels_and_coordinates(preds_bbox2d)

		mAPs = []
		for i, _ in enumerate(annotations_labels):
			# If the ground truth is healthy or there is no prediction
			if not annotations_labels[i] or not pred_labels[i]:
				continue
			mAP = metrics.calc_map(pred_bboxes[i], pred_labels[i], annotations_bboxes[i], annotations_labels[i])
			mAPs.append([mAP.map50_95, mAP.map50, mAP.map75])

		avg_mAPs = np.mean(mAPs, axis=0)

		print(f'Results for Chest X-Rays')
		print('---------------------------')
		print(f'{"Accuracy:":<21} {accuracy:.3f}')
		print(f'{"F1-Score Healthy:":<21} {f1_score_healthy:.3f}')
		print(f'{"F1-Score Unhealthy:":<21} {f1_score_unhealthy:.3f}')
		print(f'{"Average of mAP@50:95:":<21} {avg_mAPs[0]:.3f}')
		print(f'{"Average of mAP@50:":<21} {avg_mAPs[1]:.3f}')
		print(f'{"Average of mAP@75:":<21} {avg_mAPs[2]:.3f}')

		experiment_metrics = [accuracy, f1_score_healthy, f1_score_unhealthy, *avg_mAPs]
	elif experiment == 'nova_brain':
		annotations_bbox2d, annotations_desc = annotations
		preds_bbox2d, preds_desc = preds

		# Format all descriptions to lowercase
		annotations_desc_list = [value.lower() for value in annotations_desc.values()]
		preds_desc_list = [value.lower() for value in preds_desc.values()]

		bleu_scores = []
		for i, _ in enumerate(annotations_desc_list):
			bleu_score = metrics.calc_bleu([annotations_desc_list[i]], [preds_desc_list[i]])
			bleu_scores.append([bleu_score[0][0], bleu_score[0][1], bleu_score[0][2], bleu_score[0][3]])

		avg_bleu = np.mean(bleu_scores, axis=0)

		avg_llm = evaluate_with_llm(annotations_desc, preds_desc)

		mAPs = []
		for image_id in annotations_bbox2d:
			# If the ground truth is healthy or there is no prediction
			if not annotations_bbox2d[image_id] or not preds_bbox2d[image_id]:
				continue
			mAP = metrics.calc_map(preds_bbox2d[image_id], [0] * len(list(preds_bbox2d[image_id])), annotations_bbox2d[image_id], [0] * len(list(annotations_bbox2d[image_id])))
			mAPs.append([mAP.map50_95, mAP.map50, mAP.map75])

		avg_mAPs = np.mean(mAPs, axis=0)

		print(f'Results for MRI Brain Slices')
		print('--------------------------------')
		print(f'{"Average of BLEU-1:":<26} {avg_bleu[0]:.3f}')
		print(f'{"Average of BLEU-2:":<26} {avg_bleu[1]:.3f}')
		print(f'{"Average of BLEU-3:":<26} {avg_bleu[2]:.3f}')
		print(f'{"Average of BLEU-4:":<26} {avg_bleu[3]:.3f}')
		print(f'{"Average of LLM Evaluation:":<26} {avg_llm:.3f}')
		print(f'{"Average of mAP@50:95:":<26} {avg_mAPs[0]:.3f}')
		print(f'{"Average of mAP@50:":<26} {avg_mAPs[1]:.3f}')
		print(f'{"Average of mAP@75:":<26} {avg_mAPs[2]:.3f}')

		experiment_metrics = [*avg_bleu, avg_llm, *avg_mAPs]

	return experiment_metrics


def process_gpt_outputs(gpt_outputs, names_and_images, experiment):
	create_preds_csv(gpt_outputs, names_and_images, experiment)

	annotations_path = os.path.join('..', 'datasets', experiment, 'annotations.json')
	preds_path = os.path.join('..', 'outputs', experiment, 'preds.csv')

	annotations, preds = get_annotations_and_preds(annotations_path, preds_path, experiment)

	experiment_metrics = calc_metrics(annotations, preds, experiment)
	create_metrics_csv(experiment_metrics, experiment)
	draw_bboxes(annotations[0], preds[0], experiment)


if __name__ == '__main__':
	"""
	a, p = get_annotations_and_preds('../datasets/chest_xrays/annotations.json', '../outputs/chest_xrays/preds.csv', 'chest_xrays')
	e_m = calc_metrics(a, p, 'chest_xrays')
	create_metrics_csv(e_m, 'chest_xrays')
	draw_bboxes(a[0], p[0], 'chest_xrays')
	"""

	a, p = get_annotations_and_preds('../datasets/nova_brain/annotations.json', '../outputs/nova_brain/preds.csv','nova_brain')
	e_m = calc_metrics(a, p, 'nova_brain')
	create_metrics_csv(e_m, 'nova_brain')
	draw_bboxes(a[0], p[0], 'nova_brain')
