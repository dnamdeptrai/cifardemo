# CIFAR-10 ResNet-18 — Streamlit

This app performs inference with the trained checkpoint; it does not train the
model. Keep `best_cifar10_resnet18.pt` in the same folder as `app.py`. The
checkpoint is loaded once and cached for reuse.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Put `app.py`, `requirements.txt`, and `best_cifar10_resnet18.pt` at the root
   of a GitHub repository.
2. In Streamlit Community Cloud, create an app from that GitHub repository and
   select `app.py` as the entry point.

The first startup loads the checkpoint and initializes the model. The
`st.cache_resource` cache reuses that model for subsequent image predictions.
If the app process is restarted, it loads the local checkpoint again, but never
re-trains the model.
