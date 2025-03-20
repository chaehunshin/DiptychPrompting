import torch
import torch.nn.functional as F

from diffusers.utils import load_image, check_min_version
from controlnet_flux import FluxControlNetModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
import os
import numpy as np
from PIL import Image
import argparse

from diffusers.models.attention_processor import Attention

from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
import cv2
from transformers import AutoProcessor, pipeline, AutoModelForMaskGeneration

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

def detect(
    object_detector,
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
    detector_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    # object_detector = detect_pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

    labels = [label if label.endswith(".") else label+"." for label in labels]

    results = object_detector(image,  candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results

def segment(
    segmentator,
    processor,
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

def grounded_segmentation(
    detect_pipeline,
    segmentator,
    segment_processor,
    image: Union[Image.Image, str],
    labels: List[str],
    threshold: float = 0.3,
    polygon_refinement: bool = False,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None
) -> Tuple[np.ndarray, List[DetectionResult]]:
    if isinstance(image, str):
        image = load_image(image)

    detections = detect(detect_pipeline, image, labels, threshold, detector_id)
    detections = segment(segmentator, segment_processor, image, detections, polygon_refinement)

    return np.array(image), detections


class CustomFluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, height=44, width=88, attn_enforce=1.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.height = height
        self.width = width
        self.num_pixels = height * width
        self.step = 0
        self.attn_enforce = attn_enforce

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        self.step += 1
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)


        ######### attn_enforce
        if self.attn_enforce != 1.0:
            attn_probs = (torch.einsum('bhqd,bhkd->bhqk', query, key) * attn.scale).softmax(dim=-1)
            img_attn_probs = attn_probs[:, :, -self.num_pixels:, -self.num_pixels:]
            img_attn_probs = img_attn_probs.reshape((batch_size, attn.heads, self.height, self.width, self.height, self.width))
            img_attn_probs[:, :, :, self.width//2:, :, :self.width//2] *= self.attn_enforce
            img_attn_probs = img_attn_probs.reshape((batch_size, attn.heads, self.num_pixels, self.num_pixels))
            attn_probs[:, :, -self.num_pixels:, -self.num_pixels:] = img_attn_probs
            hidden_states = torch.einsum('bhqk,bhkd->bhqd', attn_probs, value)
        else:
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attn_enforce', type=float, default=1.3)
    parser.add_argument('--ctrl_scale', type=float, default=0.95)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--pixel_offset', type=int, default=8)
    parser.add_argument('--input_image_path', type=str, default='./assets/bear_plushie.jpg')
    parser.add_argument('--subject_name', type=str, default='bear plushie')
    parser.add_argument('--target_prompt', type=str, default='a photo of a bear plushie surfing on the beach')

    args = parser.parse_args()

    # Build pipeline
    controlnet = FluxControlNetModel.from_pretrained("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", torch_dtype=torch.bfloat16)
    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.transformer.to(torch.bfloat16)
    pipe.controlnet.to(torch.bfloat16)
    base_attn_procs = pipe.transformer.attn_processors.copy()

    detector_id = "IDEA-Research/grounding-dino-tiny"
    segmenter_id = "facebook/sam-vit-base"

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).cuda()
    segment_processor = AutoProcessor.from_pretrained(segmenter_id)
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=torch.device("cuda"))

    def segment_image(image, object_name):
        image_array, detections = grounded_segmentation(
            object_detector,
            segmentator,
            segment_processor,
            image=image,
            labels=object_name,
            threshold=0.3,
            polygon_refinement=True,
        )
        segment_result = image_array * np.expand_dims(detections[0].mask / 255, axis=-1) + np.ones_like(image_array) * (
                1 - np.expand_dims(detections[0].mask / 255, axis=-1)) * 255
        segmented_image = Image.fromarray(segment_result.astype(np.uint8))
        return segmented_image


    def make_diptych(image):
        ref_image = np.array(image)
        ref_image = np.concatenate([ref_image, np.zeros_like(ref_image)], axis=1)
        ref_image = Image.fromarray(ref_image)
        return ref_image


    # Load image and mask
    width = args.width + args.pixel_offset * 2
    height = args.height + args.pixel_offset * 2
    size = (width*2, height)

    subject_name = args.subject_name
    base_prompt = f"a photo of {subject_name}"
    target_prompt = args.target_prompt
    diptych_text_prompt = f"A diptych with two side-by-side images of same {subject_name}. On the left, {base_prompt}. On the right, replicate this {subject_name} exactly but as {target_prompt}"

    reference_image = load_image(args.input_image_path).resize((width, height)).convert("RGB")

    ctrl_scale=args.ctrl_scale
    segmented_image = segment_image(reference_image, subject_name)
    mask_image = np.concatenate([np.zeros((height, width, 3)), np.ones((height, width, 3))*255], axis=1)
    mask_image = Image.fromarray(mask_image.astype(np.uint8))
    diptych_image_prompt = make_diptych(segmented_image)

    new_attn_procs = base_attn_procs.copy()
    for i, (k, v) in enumerate(new_attn_procs.items()):
        new_attn_procs[k] = CustomFluxAttnProcessor2_0(height=height // 16, width=width // 16 * 2, attn_enforce=args.attn_enforce)
    pipe.transformer.set_attn_processor(new_attn_procs)

    generator = torch.Generator(device="cuda").manual_seed(42)
    # Inpaint
    result = pipe(
        prompt=diptych_text_prompt,
        height=size[1],
        width=size[0],
        control_image=diptych_image_prompt,
        control_mask=mask_image,
        num_inference_steps=30,
        generator=generator,
        controlnet_conditioning_scale=ctrl_scale,
        guidance_scale=3.5,
        negative_prompt="",
        true_guidance_scale=3.5
    ).images[0]

    result = result.crop((width, 0, width*2, height))
    result = result.crop((args.pixel_offset, args.pixel_offset, width-args.pixel_offset, height-args.pixel_offset))
    result.save('result.png')
