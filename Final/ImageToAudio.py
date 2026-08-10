import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from os import path as os_path, makedirs as os_makedirs
from time import time
from collections import deque, Counter
from sys import path as sys_path
from gtts import gTTS

project_root = os_path.dirname(os_path.dirname(os_path.abspath("Model.py")))
sys_path.append(project_root)
from Model_Training.Model import CNN

IMAGE_SIZE = 64
BOX_SIZE = 300

best_model_path = "../models/best.pt"
last_model_path = "../models/last.pt"

dataset_classes = sorted(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                     'del', 'nothing', 'space'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN(num_classes=29).to(device)
model.load_state_dict(torch.load(best_model_path))
model.eval()

inference_transforms = transforms.Compose([
    transforms.ToPILImage(),                 # ۱. تبدیل آرایه به PIL (امنتر)
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),                   # ۲. تبدیل به تنسور و مقیاس [0,1]
])

def text_to_audio(x: str, addr: str, name: str) -> None:
        """
        Convert a text string to speech and save it as an audio file.

        Args:
                x (str): The text to be spoken.
                addr (str): Directory path where the audio file will be saved.
                name (str): Filename (with or without extension) for the audio file.
        """
        # Ensure the target directory exists; create it if necessary.
        os_makedirs(addr, exist_ok=True)

        # Build the full file path.
        file_path = os_path.join(addr, name)

        # If no common audio extension is provided, append '.mp3'.
        if not file_path.lower().endswith(('.mp3', '.wav')):
                file_path += '.mp3'

        # Create a gTTS object and save the audio.
        tts = gTTS(text=x, lang='en')   # language can be changed if needed
        tts.save(file_path)
        
# Function to draw the box and prediction results on the frame
def draw_box_and_probs(frame, probs):
        h, w, _ = frame.shape

        # --- 1. Calculate the central square box coordinates ---
        center_x, center_y = w // 2, h // 2
        half = BOX_SIZE // 2
        x1 = max(0, center_x - half)
        y1 = max(0, center_y - half)
        x2 = min(w, center_x + half)
        y2 = min(h, center_y + half)

        # --- 2. Draw the box on the original frame (bright green) ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # --- 3. (Optional) Show a small thumbnail of the cropped region in the corner ---
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
                roi_resized = cv2.resize(roi, (100, 100))
                frame[10:110, w-110:w-10] = roi_resized          # top‑right corner
                cv2.putText(frame, "Input", (w-110, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # --- 4. Show top‑3 classes on the image ---
        top_indices = np.argsort(probs)[0][::-1][:5]
        top_probs = probs[0][top_indices]

        y_offset = 30
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
                label = dataset_classes[idx]
                text = f"{label}: {prob*100:.1f}%"
                color = (0, 255, 0) if i == 0 else (0, 200, 200)
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2, cv2.LINE_AA)
                y_offset += 40

        # Small instruction for the user
        cv2.putText(frame, "Show your hand in the green border", (10, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame, (x1, y1, x2, y2)

frame_timestamps_2s = deque()
high_preds_2s = deque()
frame_timestamps_5s = deque()
high_preds_5s = deque()

last_selection_time = time()
recognized_string = ""
prob_threshold = 0.75

cap = cv2.VideoCapture(0)
if not cap.isOpened():
        print("Error: Camera not found! (Try cap = cv2.VideoCapture(1) or 2)")
        exit()
        
while True:
        ret, frame = cap.read()
        if not ret:
                break

        h, w, _ = frame.shape
        current_time = time()

        center_x, center_y = w // 2, h // 2
        half = BOX_SIZE // 2
        x1 = max(0, center_x - half)
        y1 = max(0, center_y - half)
        x2 = min(w, center_x + half)
        y2 = min(h, center_y + half)

        cropped = frame[y1:y2, x1:x2]
        if cropped.size > 0:
                rgb_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                input_tensor = inference_transforms(rgb_cropped).unsqueeze(0).to(device)
                with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.softmax(output, dim=1).cpu().numpy()
        else:
                probs = np.zeros((1, len(dataset_classes)))

        frame_timestamps_2s.append(current_time)
        frame_timestamps_5s.append(current_time)

        for idx, prob in enumerate(probs[0]):
                if prob > prob_threshold:
                        high_preds_2s.append((current_time, idx))
                        high_preds_5s.append((current_time, idx))

        cutoff_2s = current_time - 2.0
        while frame_timestamps_2s and frame_timestamps_2s[0] < cutoff_2s:
                frame_timestamps_2s.popleft()
        while high_preds_2s and high_preds_2s[0][0] < cutoff_2s:
                high_preds_2s.popleft()

        cutoff_5s = current_time - 5.0
        while frame_timestamps_5s and frame_timestamps_5s[0] < cutoff_5s:
                frame_timestamps_5s.popleft()
        while high_preds_5s and high_preds_5s[0][0] < cutoff_5s:
                high_preds_5s.popleft()

        if current_time - last_selection_time >= 2.0:
                total_frames_2s = len(frame_timestamps_2s)
                if total_frames_2s > 0:
                        class_counts_2s = Counter(idx for _, idx in high_preds_2s)
                        
                        selected = False
                        for cls_idx, count in class_counts_2s.items():
                                if count >= 0.75 * total_frames_2s:
                                        label = dataset_classes[cls_idx]

                                        if label == "nothing":
                                                total_frames_5s = len(frame_timestamps_5s)
                                                count_nothing_5s = sum(1 for _, idx5 in high_preds_5s if idx5 == cls_idx)
                                                if total_frames_5s > 0 and count_nothing_5s >= 0.75 * total_frames_5s:
                                                        cv2.destroyAllWindows()
                                                        cap.release()
                                                        print(f"Result: {recognized_string}")
                                                        exit(0)
                                                else:
                                                        frame_timestamps_2s.clear()
                                                        high_preds_2s.clear()
                                                        last_selection_time = current_time
                                                        selected = True
                                                        break
                                        else:
                                                if label == "space":
                                                        recognized_string += ' '
                                                elif label == "del":
                                                        if len(recognized_string) > 0:
                                                                recognized_string = recognized_string[:-1]
                                                elif len(recognized_string) == 0 or recognized_string[-1] == '.':
                                                                recognized_string += label
                                                else:
                                                        recognized_string += label.lower()
                                                print(f"تشخیص داده شد: {label} (تعداد: {count} از {total_frames_2s})")
                                                
                                                frame_timestamps_2s.clear()
                                                high_preds_2s.clear()
                                                last_selection_time = current_time
                                                selected = True
                                                break

        frame, _ = draw_box_and_probs(frame, probs)

        cv2.putText(frame, f"Recognized: {recognized_string}", (10, h-50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
                break
        cv2.imshow("Recognizing sign language", frame)

cap.release()
cv2.destroyAllWindows()
print(f"Result: {recognized_string}")
text_to_audio(recognized_string, "../result/", "final")