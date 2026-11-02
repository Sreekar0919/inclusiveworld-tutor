"""
InclusiveWorld Gradio app entrypoint (Hugging Face Spaces compatible).
"""

from main import build_app


if __name__ == "__main__":
    build_app().launch()

