import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as transforms
import cv2
from skimage import color, filters, morphology, measure

class ROIDetector:
    def __init__(self, model_path=None, confidence=0.25, force_cpu=True, use_face_detection=True, use_skin_lesion_segmentation=False):
        """
        Initialize YOLO model for ROI detection.
        Args:
            model_path: Path to YOLO model, if None, will use YOLOv8n
            confidence: Confidence threshold for object detection
            force_cpu: Force model to run on CPU (avoids CUDA NMS errors)
            use_face_detection: Whether to use face detection model instead of generic object detection
            use_skin_lesion_segmentation: Whether to use skin lesion segmentation instead of object detection
        """
        self.use_skin_lesion_segmentation = use_skin_lesion_segmentation
        
        # Skip YOLO initialization if we're only using lesion segmentation
        if not use_skin_lesion_segmentation:
            # If we're using face detection and no specific model is provided
            if use_face_detection and model_path is None:
                # Use a face detection model from HuggingFace
                try:
                    from huggingface_hub import hf_hub_download
                    print("Downloading face detection model from HuggingFace...")
                    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
                    print(f"Downloaded face detection model to {model_path}")
                except Exception as e:
                    print(f"Warning: Could not download face detection model: {e}")
                    model_path = None
            
            # Initialize the model
            if force_cpu:
                # Force CPU device
                self.device = "cpu"
                self.model = YOLO(model_path if model_path else "yolov8n.pt").to(self.device)
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = YOLO(model_path if model_path else "yolov8n.pt")
                
            self.confidence = confidence
            
        self.to_tensor = transforms.ToTensor()
        self.force_cpu = force_cpu
        
    def detect_rois(self, image, expand_ratio=0.05, use_entire_image_fallback=True):
        """
        Detect regions of interest in the image
        Args:
            image: PIL Image or torch tensor
            expand_ratio: How much to expand the bounding box (percentage)
            use_entire_image_fallback: If True, use the entire image as ROI when detection fails
        Returns:
            List of ROI tensors, List of ROI coordinates
        """
        if isinstance(image, torch.Tensor):
            # Convert tensor to numpy for processing
            if image.dim() == 4:  # Batch of images
                image = image[0]  # Take first image
            if image.shape[0] == 3:  # If in CHW format
                image = image.permute(1, 2, 0)  # Convert to HWC
            image_np = ((image.clamp(-1, 1) * 0.5 + 0.5) * 255).byte().cpu().numpy()
            pil_image = Image.fromarray(image_np)
        else:
            pil_image = image
            # Convert to numpy for processing
            image_np = np.array(pil_image)
            
        # Get original dimensions
        W, H = pil_image.size if isinstance(pil_image, Image.Image) else (image_np.shape[1], image_np.shape[0])
        
        rois = []
        roi_coords = []
        
        # If using skin lesion segmentation
        if self.use_skin_lesion_segmentation:
            try:
                # Segment skin lesion
                mask, contours = self.segment_skin_lesion(image_np)
                
                if contours and len(contours) > 0:
                    # Find the largest contour (likely the main lesion)
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Expand box slightly
                    x1 = max(0, x - int(expand_ratio * w))
                    y1 = max(0, y - int(expand_ratio * h))
                    x2 = min(W, x + w + int(expand_ratio * w))
                    y2 = min(H, y + h + int(expand_ratio * h))
                    
                    # Extract ROI
                    if isinstance(image, torch.Tensor):
                        if image.dim() == 3:  # CHW format
                            roi = image[:, y1:y2, x1:x2]
                        else:  # HWC format
                            roi = image[y1:y2, x1:x2, :]
                    else:
                        roi = pil_image.crop((x1, y1, x2, y2))
                        roi = self.to_tensor(roi)
                    
                    rois.append(roi)
                    roi_coords.append((x1, y1, x2, y2))
                else:
                    print("No lesion regions detected")
            except Exception as e:
                print(f"Warning: Skin lesion segmentation failed: {e}")
        else:
            try:
                # Run YOLO inference (on CPU if force_cpu=True)
                if self.force_cpu:
                    # Make sure image is on CPU
                    pil_image_cpu = pil_image
                    results = self.model(pil_image_cpu, conf=self.confidence, device=self.device)
                else:
                    results = self.model(pil_image, conf=self.confidence)
                
                # Process results
                if results and len(results) > 0:
                    for result in results:
                        boxes = result.boxes
                        
                        if len(boxes) > 0:  # If we found any boxes
                            for i, box in enumerate(boxes):
                                # Get coordinates
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                
                                # Expand box slightly
                                width, height = x2 - x1, y2 - y1
                                x1 = max(0, x1 - expand_ratio * width)
                                y1 = max(0, y1 - expand_ratio * height)
                                x2 = min(W, x2 + expand_ratio * width)
                                y2 = min(H, y2 + expand_ratio * height)
                                
                                # Convert to integers
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                # Extract ROI
                                if isinstance(image, torch.Tensor):
                                    if image.dim() == 3:  # CHW format
                                        roi = image[:, y1:y2, x1:x2]
                                    else:  # HWC format
                                        roi = image[y1:y2, x1:x2, :]
                                else:
                                    roi = pil_image.crop((x1, y1, x2, y2))
                                    roi = self.to_tensor(roi)
                                
                                rois.append(roi)
                                roi_coords.append((x1, y1, x2, y2))
            except Exception as e:
                print(f"Warning: YOLO ROI detection failed: {e}")
        
        # If no ROIs were found and we're using the fallback, use the entire image
        if (not roi_coords) and use_entire_image_fallback:
            print("No ROIs detected, using entire image as ROI")
            x1, y1, x2, y2 = 0, 0, W, H
            
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:  # CHW format
                    roi = image.clone()
                else:  # HWC format
                    roi = image.clone()
            else:
                roi = self.to_tensor(pil_image)
                
            rois.append(roi)
            roi_coords.append((x1, y1, x2, y2))
        
        return rois, roi_coords

    def segment_skin_lesion(self, image):
        """
        Segment skin lesion in an image using simple image processing
        
        Args:
            image: Image as numpy array (RGB)
            
        Returns:
            Binary mask, contours
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            # Convert to LAB color space (better for skin lesion segmentation)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Extract the A channel (redness) which typically highlights lesions well
            gray = lab[:, :, 1] 
        else:
            gray = image
            
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Dilate to get a better mask
        mask = cv2.dilate(opening, kernel, iterations=3)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return mask, contours

    def apply_rois_to_mask(self, image_size, roi_coords):
        """
        Create a binary mask where ROI areas are 1 and the rest is 0
        Args:
            image_size: Tuple (H, W) or (C, H, W)
            roi_coords: List of ROI coordinates (x1, y1, x2, y2)
        Returns:
            Binary mask as tensor
        """
        if len(image_size) == 3:
            _, H, W = image_size
        else:
            H, W = image_size
            
        mask = torch.zeros((H, W), dtype=torch.float32)
        
        for x1, y1, x2, y2 in roi_coords:
            mask[y1:y2, x1:x2] = 1.0
            
        return mask

    def create_detailed_lesion_mask(self, image):
        """
        Create a detailed segmentation mask for skin lesions
        This method provides a more precise mask than simple bounding boxes
        
        Args:
            image: Image as tensor or PIL image
            
        Returns:
            Binary mask as tensor where lesion pixels are 1
        """
        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # Batch of images
                image = image[0]  # Take first image
            if image.shape[0] == 3:  # If in CHW format
                image = image.permute(1, 2, 0)  # Convert to HWC
            image_np = ((image.clamp(-1, 1) * 0.5 + 0.5) * 255).byte().cpu().numpy()
        else:
            image_np = np.array(image)
            
        # Call segmentation method
        mask_np, _ = self.segment_skin_lesion(image_np)
        
        # Convert to tensor
        mask = torch.from_numpy(mask_np.astype(np.float32) / 255.0)
        
        return mask 