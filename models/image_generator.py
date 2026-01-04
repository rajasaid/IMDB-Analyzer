import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
from typing import List, Optional, Union
import logging

class ImageGenerator:
    """
    Wrapper class for SDXL-Turbo fast image generation.
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/sdxl-turbo",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
        use_safetensors: bool = True
    ):
        """
        Initialize the ImageGenerator with SDXL-Turbo model.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            torch_dtype: Data type for model weights (float16 recommended for GPU)
            use_safetensors: Whether to use safetensors format
        """
        self.model_id = model_id
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.torch_dtype = torch_dtype if self.device == 'cuda' else torch.float32
        
        logging.info(f"Loading SDXL-Turbo model on {self.device}...")
        
        # Load the pipeline - SDXL-Turbo uses AutoPipelineForText2Image
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            variant="fp16" if self.torch_dtype == torch.float16 else None
        )
        
        self.pipeline.to(self.device)
        
        # Optional: Enable memory optimizations
        if self.device == 'cuda':
            self.pipeline.enable_attention_slicing()
            # Uncomment if you have limited VRAM:
            # self.pipeline.enable_vae_slicing()
            # self.pipeline.enable_model_cpu_offload()
        
        logging.info("SDXL-Turbo model loaded successfully!")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 1,
        guidance_scale: float = 0.0,
        seed: Optional[int] = None
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Generate images from a text prompt using SDXL-Turbo.
        
        Args:
            prompt: Text description of the image to generate
            negative_prompt: Things to avoid (NOTE: less effective with SDXL-Turbo)
            num_images: Number of images to generate
            height: Image height in pixels (must be divisible by 8, recommended 512)
            width: Image width in pixels (must be divisible by 8, recommended 512)
            num_inference_steps: Number of denoising steps (1-4 for SDXL-Turbo)
            guidance_scale: Should be 0.0 for SDXL-Turbo (it's trained for guidance_scale=0)
            seed: Random seed for reproducibility (None for random)
        
        Returns:
            Single PIL Image if num_images=1, otherwise list of PIL Images
        """
        # SDXL-Turbo specific: guidance_scale should be 0.0
        if guidance_scale != 0.0:
            logging.warning("SDXL-Turbo is optimized for guidance_scale=0.0, overriding provided value")
            guidance_scale = 0.0
        
        # SDXL-Turbo specific: 1-4 steps only
        if num_inference_steps > 4:
            logging.warning(f"SDXL-Turbo works best with 1-4 steps, you specified {num_inference_steps}")
        
        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        logging.info(f"Generating {num_images} image(s) with prompt: '{prompt[:50]}...'")
        
        # Generate images
        with torch.inference_mode():
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )
        
        images = output.images
        
        # Return single image or list
        return images[0] if num_images == 1 else images
    
    def generate_cinematic(
        self,
        base_prompt: str,
        mood: str = "dramatic",
        num_inference_steps: int = 2,
        **kwargs
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Generate cinematic-style images with enhanced prompting.
        
        Args:
            base_prompt: Base description of the scene
            mood: Cinematic mood (dramatic, melancholic, tense, joyful, etc.)
            num_inference_steps: 1-4 steps (2-4 recommended for better quality)
            **kwargs: Additional arguments passed to generate()
        
        Returns:
            Generated image(s)
        """
        # Enhance prompt with cinematic keywords
        enhanced_prompt = (
            f"{base_prompt}, cinematic still, film grain, 35mm, "
            f"{mood} lighting, professional color grading, "
            f"depth of field, cinematic composition, movie scene, high quality"
        )
        
        # Note: negative prompts are less effective with SDXL-Turbo but can still help
        negative_prompt = (
            "cartoon, anime, illustration, painting, drawing, "
            "low quality, blurry, distorted"
        )
        
        return self.generate(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            **kwargs
        )
    
    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[Image.Image]:
        """
        Generate multiple images from different prompts.
        
        Args:
            prompts: List of text prompts
            **kwargs: Additional arguments passed to generate()
        
        Returns:
            List of generated PIL Images
        """
        images = []
        for prompt in prompts:
            image = self.generate(prompt, num_images=1, **kwargs)
            images.append(image)
        return images
    
    def clear_cache(self):
        """Clear CUDA cache to free up memory."""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            logging.info("CUDA cache cleared")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize generator with SDXL-Turbo
    generator = ImageGenerator()
    
    # Example 1: Ultra-fast generation (1 step)
    prompt = "A dark empty movie theater, moody blue lighting, atmospheric, cinematic"
    image = generator.generate(prompt, num_inference_steps=1)
    image.save("movie_theater_turbo.png")
    print("Image saved as movie_theater_turbo.png (1 step, ~2 seconds)")
    
    # Example 2: Better quality (4 steps)
    prompt = "Lonely protagonist walking through rain-soaked city streets at night"
    image = generator.generate(prompt, num_inference_steps=4, seed=42)
    image.save("rainy_street_turbo.png")
    print("Image saved as rainy_street_turbo.png (4 steps, ~5 seconds)")
    
    # Example 3: Cinematic generation for movie review
    review_prompt = "tense thriller scene, shadows and dramatic lighting"
    image = generator.generate_cinematic(
        base_prompt=review_prompt,
        mood="tense",
        num_inference_steps=2,
        height=512,
        width=512
    )
    image.save("cinematic_scene_turbo.png")
    print("Cinematic image saved as cinematic_scene_turbo.png (2 steps)")
    
    # Example 4: Generate multiple variations quickly
    images = generator.generate(
        prompt="Horror movie scene, eerie atmosphere, dark shadows",
        num_images=3,
        num_inference_steps=1
    )
    for i, img in enumerate(images):
        img.save(f"horror_variation_turbo_{i}.png")
    print(f"Generated {len(images)} variations in seconds")
    
    # Example 5: Batch generation with different prompts
    prompts = [
        "Happy romantic comedy scene, bright colors, cheerful",
        "Action movie explosion, dynamic, intense",
        "Sci-fi movie spaceship interior, futuristic"
    ]
    images = generator.generate_batch(prompts, num_inference_steps=2)
    for i, img in enumerate(images):
        img.save(f"batch_image_{i}.png")
    print(f"Generated {len(images)} different scenes")
    
    # Clean up
    generator.clear_cache()