# CIFAR-10 ResNet-18 — Streamlit

This app performs inference with the trained checkpoint; it does not train the
model. It downloads the checkpoint from the public Hugging Face model
repository `dnamdeptrai/cifar` and caches the loaded model for reuse.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Put `app.py` and `requirements.txt` at the root of a GitHub repository.
2. Make sure `dnamdeptrai/cifar` is public and contains
   `cifar10-hf-space/best_cifar10_resnet18.pt`.
3. In Streamlit Community Cloud, create an app from that GitHub repository and
   select `app.py` as the entry point.

The first startup downloads the checkpoint and initializes the model. The
`st.cache_resource` cache reuses that model for subsequent image predictions.
If the app process is restarted, it downloads and loads the checkpoint again,
but never retrains the model.
