from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms as T


CHECKPOINT_PATH = Path(__file__).resolve().parent / "best_cifar10_resnet18.pt"
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TRANSFORM = T.Compose(
    [
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


@st.cache_resource
def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {CHECKPOINT_PATH}")

    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint phải chứa state dictionary.")

    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("Không tìm thấy state dictionary hợp lệ trong checkpoint.")

    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }

    class_names = checkpoint.get("class_names", checkpoint.get("classes", CLASS_NAMES))
    if len(class_names) != len(CLASS_NAMES):
        raise ValueError(f"Checkpoint phải có đúng {len(CLASS_NAMES)} lớp.")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, list(class_names)


@torch.inference_mode()
def predict(image, model, class_names, top_k):
    image_tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0)
    probabilities = torch.softmax(model(image_tensor), dim=1)[0]
    top_probabilities, top_indices = probabilities.topk(top_k)
    return [
        {
            "Lớp": class_names[int(index)],
            "Xác suất": float(probability),
        }
        for probability, index in zip(top_probabilities, top_indices)
    ]


st.set_page_config(page_title="CIFAR-10 ResNet-18", page_icon="🖼️")
st.title("Nhận dạng ảnh CIFAR-10")
st.write(
    "Tải ảnh lên để xem các lớp dự đoán và xác suất. "    
)

try:
    with st.spinner("Đang nạp checkpoint và khởi tạo mô hình lần đầu..."):
        model, class_names = load_model()
except Exception as error:
    st.error(f"Không thể nạp mô hình: {error}")
    st.info("Đảm bảo `best_cifar10_resnet18.pt` được đặt cạnh `app.py` trong GitHub repo.")
    st.stop()

uploaded_file = st.file_uploader(
    "Chọn ảnh", type=["png", "jpg", "jpeg", "webp"]
)
top_k = st.slider("Số lớp cần hiển thị", min_value=1, max_value=10, value=5)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh đầu vào", use_container_width=True)

    if st.button("Dự đoán", type="primary"):
        with st.spinner("Đang phân loại ảnh..."):
            results = predict(image, model, class_names, top_k)
        st.subheader("Kết quả")
        st.dataframe(results, hide_index=True, use_container_width=True)
        st.bar_chart(
            {item["Lớp"]: item["Xác suất"] for item in results},
            y_label="Xác suất",
        )
