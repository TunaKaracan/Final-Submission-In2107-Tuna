import base64
import os
import glob


def encode_images(img_folder_path):
	img_paths = glob.glob(os.path.join(img_folder_path, '*.png'))
	names_and_images = {}

	for path in img_paths:
		with open(path, 'rb') as file:
			names_and_images[path.split(os.sep)[-1].split('.')[0]] = base64.b64encode(file.read()).decode('utf-8')

	return names_and_images


def get_images(main_folder_path):
	names_and_images = encode_images(os.path.join(main_folder_path, 'images'))

	return names_and_images


if __name__ == "__main__":
	get_images(os.path.join('..', 'datasets', 'chest_xrays'))
