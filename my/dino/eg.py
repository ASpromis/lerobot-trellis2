from transformers import pipeline
from transformers.image_utils import load_image

import debugpy
debugpy.listen(4071)
print("Waiting for debugger to attach...")
debugpy.wait_for_client()

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = load_image(url)

feature_extractor = pipeline(
    model="facebook/dinov2-small",
    task="image-feature-extraction", 
)
features = feature_extractor(image)
