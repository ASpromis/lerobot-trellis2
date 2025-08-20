from huggingface_hub import HfApi

hub_api = HfApi()
hub_api.create_tag("imManjusaka/cube_placement", tag="v2.0", repo_type="dataset")