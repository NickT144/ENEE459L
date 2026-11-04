import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import warnings
from scipy.ndimage import convolve, gaussian_filter

MODEL_PATH = "version-RFB-320.onnx"

warnings.filterwarnings("ignore", category=UserWarning)

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3  

# 3. Use only available providers to prevent the provider fallback warning
desired_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
available_providers = [p for p in desired_providers if p in ort.get_available_providers()]

# face detection
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores, boxes = box_scores
    box_indices = np.argsort(scores)
    box_indices = list(box_indices[-candidate_size:])
    picked = list()

    while len(box_indices) > 0:
        index = box_indices.pop()
        curr = boxes[index]
        picked.append(index)

        if len(picked) == top_k or len(box_indices) == 0:
            break

        x1curr, y1curr, x2curr, y2curr = curr
        area_curr = (x2curr - x1curr) * (y2curr - y1curr)
        for i in box_indices[:]:
            candidate = boxes[i]
            x1can, y1can, x2can, y2can = candidate
            x1max = max(x1curr, x1can)
            x2min = min(x2curr, x2can)
            y1max = max(y1curr, y1can)
            y2min = min(y2curr, y2can)
            intersection = max(0, x2min - x1max) * max(0, y2min - y1max)

            area_can = (x2can - x1can) * (y2can - y1can)
            union = area_curr + area_can - intersection
            iou = intersection / union

            if iou > iou_threshold:
                box_indices.remove(i)

    return picked, boxes[picked]



def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):
    img = Image.open(image_path)
    img = img.convert("RGB")
    w, h = img.size

    resized = img.resize((320, 240))
    resized = np.array(resized, dtype=np.float32)
    resized = (resized - 127.0) / 128.0
    resized = np.transpose(resized, (2, 0, 1))
    resized = np.expand_dims(resized, axis=0)

    session = ort.InferenceSession(MODEL_PATH, providers=available_providers)
    name = session.get_inputs()[0].name
    scores, boxes = session.run(None, {name: resized})
    scores = scores[0]
    boxes = boxes[0]

    if scores.ndim == 2:
        scores = scores[:, 1]

    mask = scores >= conf_threshold
    filtered_scores = scores[mask]
    filtered_boxes = boxes[mask]

    final_indices, final_boxes = hard_nms((filtered_scores, filtered_boxes), iou_threshold=0.4)

    draw = ImageDraw.Draw(img)
    
    for index, box in zip(final_indices, final_boxes):
        x1_norm, y1_norm, x2_norm, y2_norm = box
        score = filtered_scores[index]

        x1 = x1_norm * w
        y1 = y1_norm * h
        x2 = x2_norm * w
        y2 = y2_norm * h
        
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 12)), f"Face: {score:.2f}", fill="red")

    img.save(output_path)



# Edge detection
def non_maximum_suppression(magnitude, angle):
    h, w = magnitude.shape
    out = np.zeros((h, w), dtype=np.float32)

    angle_degrees = np.degrees(angle)
    angle_degrees = angle_degrees % 180

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            neighbor_one = 255.0
            neighbor_two = 255.0
            theta = angle_degrees[i, j]
            
            if (0 <= theta < 22.5) or (157.5 <= theta <= 180):
                neighbor_one = magnitude[i, j-1]
                neighbor_two = magnitude[i, j+1]
            elif (22.5 <= theta < 67.5):
                neighbor_one = magnitude[i+1, j-1]
                neighbor_two = magnitude[i-1, j+1]
            elif (67.5 <= theta < 112.5):
                neighbor_one = magnitude[i-1, j]
                neighbor_two = magnitude[i+1, j]
            elif (112.5 <= theta < 157.5):
                neighbor_one = magnitude[i-1, j-1]
                neighbor_two = magnitude[i+1, j+1]

            if (magnitude[i, j] >= neighbor_one) and (magnitude[i, j] >= neighbor_two):
                out[i, j] = magnitude[i, j]
                
    return out
    

def hysteresis_thresholding(img, low_thresh, high_thresh):
    STRONG = 255
    WEAK = 50
    SUPPRESSED = 0
    h, w = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    strong_i, strong_j = np.where(img >= high_thresh)
    weak_i, weak_j = np.where((img >= low_thresh) & (img < high_thresh))
    
    mask[strong_i, strong_j] = STRONG
    mask[weak_i, weak_j] = WEAK
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if mask[i, j] == WEAK:
                if (mask[i+1, j-1] == STRONG or mask[i+1, j] == STRONG or mask[i+1, j+1] == STRONG
                    or mask[i, j-1] == STRONG or mask[i, j+1] == STRONG
                    or mask[i-1, j-1] == STRONG or mask[i-1, j] == STRONG or mask[i-1, j+1] == STRONG):
                    mask[i, j] = STRONG
                else:
                    mask[i, j] = SUPPRESSED
                    
    return mask

def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):
    img = Image.open(image_path)
    img = img.convert('L')
    img = np.array(img, dtype=np.float32)
    img = gaussian_filter(img, sigma=1.4)

    K_x = np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ], dtype=np.float32)
    
    K_y = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ], dtype=np.float32)
    
    G_x = convolve(img, K_x)
    G_y = convolve(img, K_y)
    
    M = np.sqrt(G_x**2 + G_y**2)
    M = (M / M.max()) * 255.0
    theta = np.arctan2(G_y, G_x)
    
    thin = non_maximum_suppression(M, theta)
    
    edges = hysteresis_thresholding(thin, low, high)
    edges = edges.astype(np.uint8)
    final_img = Image.fromarray(edges)
    final_img.save(output_path)

if __name__ == "__main__":
    detect_faces("face.jpg")
    canny_scipy("face.jpg")