import comfy.model_management
import gc
from comfy.patcher_extension import CallbacksMP
from comfy.model_patcher import ModelPatcher
from comfy.model_base import WAN21

#Based on https://github.com/kijai/ComfyUI-WanVideoWrapper
class WanVideoBlockSwap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "blocks_to_swap": ("INT", {"default": 20, "min": 0, "max": 40, "step": 1, "tooltip": "Number of transformer blocks to swap, the 14B model has 40, while the 1.3B model has 30 blocks"}),
                "offload_img_emb": ("BOOLEAN", {"default": False, "tooltip": "Offload img_emb to offload_device"}),
                "offload_txt_emb": ("BOOLEAN", {"default": False, "tooltip": "Offload txt_emb to offload_device"}),
            },
        }
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "ComfyUI-wvBlockswap"
    FUNCTION = "set_callback"

    def set_callback(self, model: ModelPatcher, blocks_to_swap, offload_img_emb, offload_txt_emb):
        
        def swap_blocks(model: ModelPatcher, device_to, lowvram_model_memory, force_patch_weights, full_load):
            base_model = model.model
            if isinstance(base_model, WAN21):
                unet = base_model.diffusion_model
                for b, block in enumerate(unet.blocks):
                    if b < blocks_to_swap:
                        block.to(model.offload_device)
                if offload_img_emb:
                    unet.img_emb.to(model.offload_device)
                if offload_txt_emb:
                    unet.text_embedding.to(model.offload_device)
            
            comfy.model_management.soft_empty_cache()
            gc.collect()
        
        model = model.clone()
        model.add_callback(CallbacksMP.ON_LOAD,swap_blocks)

        return (model, )

NODE_CLASS_MAPPINGS = {
    "wvBlockSwap": WanVideoBlockSwap
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "wvBlockSwap": "WanVideoBlockSwap"
}
