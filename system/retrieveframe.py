#!/usr/bin/env python3
"""
MobileCLIP retrieval helpers: encode frames and select those matching LLM queries.
"""

from pathlib import Path
from typing import Iterable, List

import cv2
import torch
from PIL import Image
import mobileclip


def extract_frames(video_path: str, output_dir: str, every_n: int = 10) -> List[str]:
	"""
	Save every-nth frame from the video to disk; return list of frame paths.
	"""
	Path(output_dir).mkdir(parents=True, exist_ok=True)
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		return []

	frame_paths: List[str] = []
	idx = 0
	saved = 0
	while cap.isOpened():
		success, frame = cap.read()
		if not success:
			break
		if idx % every_n == 0:
			frame_file = Path(output_dir) / f"frame_{saved:05d}.jpg"
			cv2.imwrite(str(frame_file), frame)
			frame_paths.append(str(frame_file))
			saved += 1
		idx += 1

	cap.release()
	return frame_paths


def get_best_frames(frame_paths: Iterable[str], llm_query: str, pretrained_path: str) -> List[str]:
	"""
	Return top-5 frame paths most similar to the LLM-provided query using MobileCLIP.
	"""
	model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s1', pretrained=pretrained_path)
	tokenizer = mobileclip.get_tokenizer('mobileclip_s1')

	images = torch.stack([preprocess(Image.open(p).convert('RGB')) for p in frame_paths])
	with torch.no_grad():
		image_features = model.encode_image(images)
		image_features /= image_features.norm(dim=-1, keepdim=True)

	query = tokenizer([llm_query])
	with torch.no_grad():
		text_features = model.encode_text(query)
		text_features /= text_features.norm(dim=-1, keepdim=True)

	scores = image_features @ text_features.T
	topk = scores.squeeze().topk(5)
	return [list(frame_paths)[i] for i in topk.indices]
