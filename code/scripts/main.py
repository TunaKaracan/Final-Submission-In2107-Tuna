import os
import textwrap
from time import sleep

from openai import OpenAI

import image_prepare
import utils


if __name__ == "__main__":
	# Experiment should be either 'chest_xrays' or 'nova_brain'
	EXPERIMENT = 'nova_brain'
	ANNOTATIONS_PATH = os.path.join('..', 'datasets', EXPERIMENT, 'annotations.json')
	PREDS_PATH = os.path.join('..', 'outputs', EXPERIMENT, 'preds.csv')

	SYSTEM_PROMPT_XRAY = textwrap.dedent("""
		You are a board-certified radiologist and you are given a 1024x1024 chest X-ray.
		Examine this X-ray and determine if there are any signs of the following diseases: ["Aortic enlargement", "Cardiomegaly", "Pulmonary fibrosis", "Mass", "Pleural effusion", "Rib fracture", "Other lesion", "Infiltration", "Lung Opacity", "Consolidation", "Calcification", "ILD", "Pleural thickening", "Pneumonia", "Emphysema", "Pneumothorax", "Lung cyst"]. Multiple diseases could be present in the X-ray.
		If none of the diseases are present then the patient is considered healthy, otherwise unhealthy.
		You are also tasked with naming the disease(s) and locating it(them) using bounding box coordinates.
		The bounding box is structured as [x_top, y_top, x_bottom, y_bottom, disease_name] and in image coordinates (0, 0) refers to the top left and (1023, 1023) refers to the bottom right.
		The output should consist of only the diagnosis and the bounding boxes.
		An example output would be:

		Diagnosis: Healthy
		Bounding boxes: None
		or
		Diagnosis: Unhealthy
		Bounding boxes: [[120, 75, 280, 455, Infiltration], [530, 215, 690, 375, Cardiomegaly]]

		The output format should be plain text, no bold, italic texts or emojis.
		You are free to use any tool you believe will help you with this task and you are encouraged to form long chains-of-thoughts.
		An example chain of though would be:
		
		"I observe increased radiopacity in the lower left lung field consistent with consolidation. The heart size appears within normal limits. No obvious rib fractures. Therefore, Diagnosis: Unhealthy. Bounding boxes: [[250, 600, 450, 750, Consolidation]]"
		""")

	SYSTEM_PROMPT_NOVA = textwrap.dedent("""
		You are a board-certified radiologist and you are given a 1024x1024 MRI brain slice.
		Examine this MRI slice and describe it.
		You are also tasked with finding any abnormal regions and output their bounding box coordinates.
		The bounding box is structured as [x_top, y_top, x_bottom, y_bottom] and in image coordinates (0, 0) refers to the top left and (1023, 1023) refers to the bottom right.
		The output should consist of only the description and the bounding boxes.
		An example output would be:

		Description: Axial T2 gradient image shows a small hypointense area within the intraventricular mass, compatible with calcification.
		Bounding boxes: [[530, 215, 690, 375]]
		or
		Description: The MRI brain slice appears to be predominantly isointense to gray matter. There are no significant abnormal hyperintense or hypointense lesions noted.
		Bounding boxes: None

		The output format should be plain text, no bold, italic texts or emojis.
		You are free to use any tool you believe will help you with this task and you are encouraged to form long chains-of-thoughts.
		An example chain of though would be:
		
		"I observe a well-defined, predominantly hyperintense intraventricular lesion. Within this mass, a focal area of signal void is observed, suggesting a dense component. Given its hypointensity on T2, the appearance is most consistent with calcific material."
		""")

	SYSTEM_PROMPT = SYSTEM_PROMPT_XRAY if EXPERIMENT == 'chest_xrays' else SYSTEM_PROMPT_NOVA

	# The API Key is outside the project scope for safety
	API_KEY_PATH = os.path.join('..', '..', '..', 'API_KEY.txt')
	with open(API_KEY_PATH, 'r') as api_key_file:
		client = OpenAI(api_key=api_key_file.readline().strip())

	names_and_images = image_prepare.get_images(os.path.join('..', 'datasets', EXPERIMENT))
	gpt_outputs = []

	for name in names_and_images:
		user_prompt = utils.create_user_prompt(names_and_images[name])
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
		print(gpt_output)
		# Sleep for 10 seconds before next prompt to not hit the per minute token limit
		if name != list(names_and_images.keys())[-1]:
			sleep(10)

	utils.process_gpt_outputs(gpt_outputs, names_and_images, EXPERIMENT)
