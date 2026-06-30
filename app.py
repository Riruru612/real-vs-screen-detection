import time

import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms


st.set_page_config(
    page_title="Spot the Fake Photo",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    div[data-testid="metric-container"]{
        border:1px solid #d9d9d9;
        padding:15px;
        border-radius:12px;
    }

    footer{
        visibility:hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@st.cache_resource
def load_model():

    model = models.resnet34(weights=None)

    model.fc = nn.Linear(512, 2)

    model.load_state_dict(
        torch.load(
            "models/final_resnet34_detector.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    # Warmup
    dummy = torch.randn(1, 3, 224, 224).to(device)

    with torch.no_grad():
        model(dummy)

    return model


model = load_model()


def predict(image):

    img1 = transform(image)

    img2 = transform(
        image.transpose(Image.FLIP_LEFT_RIGHT)
    )

    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():

        out1 = model(img1)
        out2 = model(img2)

        prob1 = torch.softmax(out1, dim=1)[0, 1]
        prob2 = torch.softmax(out2, dim=1)[0, 1]

        score = ((prob1 + prob2) / 2).item()

    if device.type == "cuda":
        torch.cuda.synchronize()

    latency = (time.perf_counter() - start) * 1000

    return score, latency


with st.sidebar:

    st.title("Spot the Fake Photo")

    st.caption(
        "Real vs Screen Recapture Detection"
    )

    st.divider()

    st.markdown("### Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    st.divider()

    st.markdown("### Model")

    st.write("**Architecture:** ResNet34")

    st.write("**Transfer Learning:** ImageNet")

    st.write("**Input Resolution:** 224 × 224")

    st.divider()

    st.markdown("### Evaluation")

    st.write("**Mean Accuracy:** 85.33%")

    st.write("**Best Fold Accuracy:** 95.00%")

    st.write("**Mean F1 Score:** 86.06%")

st.subheader("Real vs Screen Recapture Detection")

st.write(
    """
    Upload an image to determine whether it is a genuine photograph
    or a photograph captured from a digital display.
    """
)

if uploaded_file is None:

    st.info(
        "Upload an image using the sidebar to begin the prediction."
    )

else:

    image = Image.open(uploaded_file).convert("RGB")

    score, latency = predict(image)

    real_conf = (1 - score) * 100
    screen_conf = score * 100

    left, right = st.columns([1.4, 1])

    with left:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with right:

        if score >= 0.5:

            st.error("Screen Recapture Detected")

        else:

            st.success("Real Photograph")

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Fraud Score",
                f"{score:.4f}"
            )

        with c2:

            st.metric(
                "Latency",
                f"{latency:.2f} ms"
            )

        st.markdown("### Confidence")

        st.write("Real Photograph")

        st.progress(real_conf / 100)

        st.caption(f"{real_conf:.2f}%")

        st.write("Screen Recapture")

        st.progress(screen_conf / 100)

        st.caption(f"{screen_conf:.2f}%")

        st.divider()

        st.markdown("### Prediction")

        if score >= 0.5:

            st.markdown(
                """
                The uploaded image is classified as a
                **Screen Recapture**.
                """
            )

        else:

            st.markdown(
                """
                The uploaded image is classified as a
                **Real Photograph**.
                """
            )
    st.divider()

    st.markdown("## Model Performance")

    p1, p2 = st.columns(2)

    with p1:

        st.metric(
            "Mean Accuracy",
            "85.33%"
        )

        st.metric(
            "Precision",
            "82.27%"
        )

    with p2:

        st.metric(
            "Recall",
            "90.36%"
        )

        st.metric(
            "F1 Score",
            "86.06%"
        )

    st.divider()

    st.markdown("## Benchmark")

    st.dataframe(
        {
            "Model": [
                "Random Forest",
                "XGBoost",
                "MobileNetV3",
                "EfficientNet-B0",
                "ResNet18",
                "ResNet34"
            ],
            "Mean Accuracy": [
                "80.29%",
                "83.29%",
                "77.52%",
                "82.52%",
                "81.33%",
                "85.33%"
            ],
            "Best Fold": [
                "90.48%",
                "90.48%",
                "95.00%",
                "100.00%",
                "90.48%",
                "95.00%"
            ],
        },
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    i1, i2 = st.columns(2)

    with i1:

        st.markdown("### Model Details")

        st.write("Architecture : **ResNet34**")

        st.write("Transfer Learning : **ImageNet**")

        st.write("Input Size : **224 × 224**")

        st.write("Evaluation : **5-Fold Stratified Cross Validation**")

        st.write("Deployment : **PyTorch**")

    with i2:

        st.markdown("### Deployment")

        st.write("Inference : **On Device**")

        st.write("Average Latency : **30–40 ms**")

        st.write("Cost per Image : **≈ $0**")

        st.write("Cloud Required : **No**")

        st.write("Framework : **Streamlit**")

    st.divider()

    st.markdown("### About")

    st.write(
        """
        This application detects whether an uploaded image is a genuine
        photograph or a photograph captured from a digital display.
        The deployed model is a fine-tuned ResNet34 selected after
        comparing multiple classical machine learning and transfer
        learning approaches using Stratified 5-Fold Cross Validation.
        """
    )

st.divider()

c1, c2, c3 = st.columns(3)

with c1:

    st.caption("Model : ResNet34")

with c2:

    st.caption("Framework : PyTorch")

with c3:

    st.caption("Developed for the SalesCode AI Take-Home Assignment")