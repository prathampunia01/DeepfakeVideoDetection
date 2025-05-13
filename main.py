import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import gradio as gr
from facenet_pytorch import MTCNN, InceptionResnetV1
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import warnings
warnings.filterwarnings('ignore')

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

# Load the model weights
checkpt = torch.load("resnetinceptionv1_epoch_32.pth", map_location=DEVICE)
model.load_state_dict(checkpt['model_state_dict'])
model.to(DEVICE)
model.eval()

def predict(input_video):
    cap = cv2.VideoCapture(input_video)
    

    frame_results = []
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame.")
            break
        print("Processing a frame...")
        # Convert frame to PIL image
        input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Run face detection
        face = mtcnn(input_image)
        if face is None:
            print("No face detected, skipping frame...")
            continue
        face = face.unsqueeze(0)
        face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)

        prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
        prev_face = prev_face.astype('uint8')

        face = face.to(DEVICE)
        face = face.to(torch.float32)
        face = face / 255.0
        face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

        target_layers = [model.block8.branch1[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(0)]

        grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
        face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

        with torch.no_grad():
            output = torch.sigmoid(model(face).squeeze(0))
            prediction = 'real' if output.item() < 0.5 else "fake"
            real_prediction = 1 - output.item()
            fake_prediction = output.item()

            confidences = {
                'real': real_prediction,
                'fake': fake_prediction
            }

        # Append result for the current frame
        frame_results.append((confidences, face_with_mask))

    cap.release()

    # Aggregating results (you could choose to return the last frame's result, the most common prediction, etc.)
    last_confidences, last_face_with_mask = frame_results[-1] if frame_results else (None, None)
    return last_confidences, last_face_with_mask

interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Video(label="Input Video")
    ],
    outputs=[
        gr.Label(label="Class"),
        gr.Image(label="Face With Explainability", type="pil")
    ]
).launch()
