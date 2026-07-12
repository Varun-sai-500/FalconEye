from core.segmentation.sam_wrapper import SAMWrapper
from core.segmentation.clipseg_wrapper import CLIPSegWrapper

sam_service = SAMWrapper(model_id="facebook/sam-vit-base")
clipseg_service = CLIPSegWrapper(sam_wrapper=sam_service, model_id="CIDAS/clipseg-rd64-refined")
