import cv2

from paint_mask import MaskPainter
from move_mask import MaskMover
from poisson_image_editing import poisson_edit
from skimage.transform import match_histograms
from skimage.filters import unsharp_mask

#import argparse
import getopt
import sys
from os import path
import numpy as np
from color_transfer import color_transfer

def usage():
	print("Usage: python main.py [options] \n\n\
	Options: \n\
	\t-h\tPrint a brief help message and exits..\n\
	\t-s\t(Required) Specify a source image.\n\
	\t-t\t(Required) Specify a target image.\n\
	\t-m\t(Optional) Specify a mask image with the object in white and other part in black, ignore this option if you plan to draw it later.")


def split_chan(im, sz):
	b, g, r = cv2.split(im)
	bright = cv2.max(cv2.max(r, g), b)
	# dark = cv2.min(cv2.min(r, g), b)
	dc = cv2.min(cv2.min(r, g), b);
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
	dark = cv2.erode(dc, kernel)
	cv2.imwrite('dark.png', dark)
	return dark, bright, r, g, b


def get_mask(img):
	dark, bright, r, g, b = split_chan(img, 1)
	ret, imgf = cv2.threshold(np.uint8(np.abs(g-dark)), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	cv2.imwrite('r-dark.png', np.uint8(np.abs(r - dark)))
	cv2.imwrite('g-dark.png', np.uint8(np.abs(g - dark)))
	# ret, imgf = cv2.threshold(np.uint8(np.abs(r - dark)), 60, 255, cv2.THRESH_BINARY)
	# ret, imgf = cv2.threshold(np.uint8(np.abs(g - dark)), 60, 255, cv2.THRESH_BINARY)
	# cv2.imshow("target", np.abs(r-dark))
	# cv2.imshow("mask", imgf)
	imgf = imgf / 255
	# cv2.waitKey()

	return imgf


def resize(img):
	small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
	return small


if __name__ == '__main__':
	# parse command line arguments
	args = {}

	try:
		opts, _ = getopt.getopt(sys.argv[1:], "hs:t:m:p:")
	except getopt.GetoptError as err:
		# print help information and exit:
		print(err)  # will print something like "option -a not recognized"
		print("See help: main.py -h")
		exit(2)
	for o, a in opts:
		if o in ("-h"):
			usage()
			exit()
		elif o in ("-s"):
			args["source"] = a
		elif o in ("-t"):
			args["target"] = a
		elif o in ("-m"):
			args["mask"] = a
		else:
			assert False, "unhandled option"

	#
	if ("source" not in args) or ("target" not in args):
		usage()
		exit()

	#
	source = cv2.imread(args["source"])
	target = cv2.imread(args["target"])
	b,g,r = cv2.split(target)
	cv2.imwrite('red.png', r)
	cv2.imwrite('blue.png', b)
	cv2.imwrite('green.png', g)

	# source = cv2.bilateralFilter(source, 11, 81, 81)
	# cv2.imwrite('bilateral.png', source)

	# source = color_transfer(source=target, target=source)
	# source = cv2.resize(source, (0, 0), fx=0.5, fy=0.5)
	# target = cv2.resize(target, (0, 0), fx=0.5, fy=0.5)
	# for i in range(3):
	# 	source[:,:,i] = cv2.equalizeHist(source[:,:,i])
	# cv2.imwrite('equ.png', source)


	# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	# source = clahe.apply(source)
	# cv2.imwrite('clahe.png', source)

	source = cv2.fastNlMeansDenoising(source, None, 3, 7, 25)
	cv2.imwrite('denoise_nlm.png', source)
	# source = cv2.medianBlur(source, 11)
	# cv2.imwrite('denoise_median.png', source)
	#
	# source = unsharp_mask(source, radius=3, amount=3, multichannel=True)
	# source = np.uint8(source * 255)
	# cv2.imwrite('usm.png', source)


	# kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
	# source = cv2.filter2D(source, -1, kernel)
	# cv2.imwrite('sharpen.png', source)

	# source = match_histograms(source, target, multichannel=True)
	# cv2.imwrite('color_match.png', source)


	# cv2.imshow("denoised", source)
	# cv2.imshow("target", target)
	# cv2.waitKey()
	if source is None or target is None:
		print('Source or target image not exist.')
		exit()

	# if source.shape[0] > target.shape[0] or source.shape[1] > target.shape[1]:
	# 	print('Source image cannot be larger than target image.')
	# 	exit()

	# draw the mask
	mask_path = ""
	if "mask" not in args:
		print('Please highlight the object to disapparate.\n')
		mp = MaskPainter(args["source"])
		mask_path = mp.paint_mask()
	else:
		mask_path = args["mask"]

	# adjust mask position for target image
	# print('Please move the object to desired location to apparate.\n')
	# mm = MaskMover(args["target"], mask_path)
	# offset_x, offset_y, target_mask_path = mm.move_mask()

	target_mask_path = args["mask"]

	# blend
	print('Blending ...')
	# target_mask = cv2.imread(target_mask_path, cv2.IMREAD_GRAYSCALE)
	target_eye_mask = cv2.imread(target_mask_path, cv2.IMREAD_GRAYSCALE)
	offset = (128, -34)
	#
	target_mask = np.uint8(get_mask(target) * 255)
	cv2.imwrite('target_mask_dark_chan_original.png', target_mask)
	kernel = np.ones((21, 21), np.uint8)
	target_mask = cv2.dilate(target_mask, kernel, iterations=1)
	cv2.imwrite('target_mask_dark_chan.png', target_mask)
	target_mask = target_mask * target_eye_mask
	cv2.imwrite('target_mask_final.png', np.uint8(target_mask * 255))
	# offset = offset_x, offset_y

	poisson_blend_result = poisson_edit(source, target, target_mask, offset)


	cv2.imwrite(path.join(path.dirname(args["source"]), 'target_result.png'),
				poisson_blend_result)

	print('Done.\n')
