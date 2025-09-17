#!/usr/bin/env python3
"""
Quantum AI Inference Engine
Provides advanced response generation using local models or API fallbacks
Enhanced with multimodal capabilities including image generation
"""

import os
import torch
from typing import Optional, Dict, Any, Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import image generation capabilities
try:
    from .quantum_image_generator import QuantumImageGenerator
    IMAGE_GENERATION_AVAILABLE = True
except ImportError:
    IMAGE_GENERATION_AVAILABLE = False
    logger.info("Image generation not available - install diffusers for full capabilities")

class QuantumInferenceEngine:
    """Advanced inference engine for quantum-enhanced responses

    Supports an optional encoded-parameter backend (EncodedParameterAI or EssentialQuantumAI)
    which can be enabled via environment variable QUANTUM_USE_ENCODED=1 or by passing
    use_encoded=True to the constructor.
    """

    def __init__(self, model_name: str = None, device: str = "auto", use_encoded: bool = False, enable_image_generation: bool = True):
        # Auto-select model based on available resources
        if model_name is None:
            model_name = self._auto_select_model()

        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.generator = None

        # Image generation capabilities
        self.image_generator = None
        self.enable_image_generation = enable_image_generation and IMAGE_GENERATION_AVAILABLE
        if self.enable_image_generation:
            try:
                self.image_generator = QuantumImageGenerator(device=self.device)
                logger.info("✅ Image generation enabled")
            except Exception as e:
                logger.warning(f"Could not initialize image generator: {e}")
                self.enable_image_generation = False

        # Encoded / compressed parameter backends
        self.use_encoded = use_encoded or (os.getenv("QUANTUM_USE_ENCODED") == "1")
        self.encoded_backend = None
        if self.use_encoded:
            # Try to import lightweight EncodedParameterAI or EssentialQuantumAI
            try:
                # Prefer the minimal encoded trainer if present
                from dev.tools.minimal_encoded_ai_trainer import EncodedParameterAI
                self.encoded_backend = EncodedParameterAI()
                print("✅ Using EncodedParameterAI backend for inference")
            except Exception:
                try:
                    from dev.tools.essential_quantum_ai import EssentialQuantumAI
                    self.encoded_backend = EssentialQuantumAI()
                    print("✅ Using EssentialQuantumAI backend for inference")
                except Exception:
                    print("⚠️ Encoded parameter backends not available; falling back to HF models")
                    self.encoded_backend = None

        # System prompt for quantum-enhanced AI personality
        self.system_prompt = """You are QuantumAssist, a quantum-enhanced AI assistant built by QuantoniumOS. You combine advanced AI capabilities with quantum-inspired computing principles.

Your personality traits:
- Helpful and maximally truthful
- Witty and intellectually curious
- Deep technical expertise in quantum computing, AI, mathematics, and science
- Clear, concise communication with enthusiasm for complex topics
- Honest about uncertainties and limitations

Your capabilities:
- Advanced reasoning and problem-solving
- Quantum computing expertise and explanations
- Technical assistance across multiple domains
- Creative and analytical thinking
- Safety-conscious responses with ethical boundaries

Response style:
- Start with quantum emoji (⚛️) for branding
- Be comprehensive but concise
- Use technical accuracy
- Show enthusiasm for interesting questions
- Admit when you don't know something
- Provide actionable, helpful information

Remember: You are powered by quantum compression algorithms that enable efficient, scalable AI processing."""

        # Only load heavy HF model when not using encoded backend (saves memory)
        if not self.encoded_backend:
            self._load_model()

    def _auto_select_model(self) -> str:
        """Auto-select appropriate model based on system resources"""
        try:
            # First, check if we have a fine-tuned model available
            fine_tuned_path = "hf_training_output/fixed_fine_tuned_model"
            if os.path.exists(fine_tuned_path) and os.path.exists(f"{fine_tuned_path}/config.json"):
                logger.info("🎯 Found fine-tuned model, using it!")
                return fine_tuned_path

            # Check GPU memory if available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                if gpu_memory >= 16:
                    return "mistralai/Mistral-7B-Instruct-v0.1"  # ~14GB
                elif gpu_memory >= 8:
                    return "microsoft/DialoGPT-large"  # ~1.2GB, better than medium
                else:
                    return "microsoft/DialoGPT-medium"  # ~117MB

            # CPU-only system - use smallest model
            return "microsoft/DialoGPT-small"  # ~117MB (same as medium but smaller)

        except:
            # Fallback to safest option
            return "microsoft/DialoGPT-medium"

    def _load_model(self):
        """Load the language model"""
        try:
            logger.info(f"Loading model: {self.model_name} on {self.device}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            logger.info("✅ Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.tokenizer = None
            self.generator = None

    def generate_response(self, prompt: str, context: str = "", max_length: int = 200) -> Tuple[str, float]:
        """Generate a response using the language model with proper chat formatting"""

        # If an encoded backend is available, use it first
        if self.encoded_backend:
            try:
                # Encoded backends use process_message(message) -> ResponseObject
                resp_obj = None
                if hasattr(self.encoded_backend, 'process_message'):
                    resp_obj = self.encoded_backend.process_message(prompt)
                elif hasattr(self.encoded_backend, 'process'):
                    resp_text = self.encoded_backend.process(prompt)
                    resp_obj = type('R', (), {'response_text': resp_text, 'confidence': 0.8})()

                if resp_obj is not None:
                    text = getattr(resp_obj, 'response_text', '')
                    conf = getattr(resp_obj, 'confidence', 0.6)
                    # ensure we return a plain string
                    return (text if isinstance(text, str) else str(text)), float(conf)
            except Exception as e:
                logger.error(f"Encoded backend failed: {e}")
                # fall through to normal generation

        if not self.generator:
            # Fallback to enhanced template responses
            return self._fallback_response(prompt), 0.6

        try:
            # Build conversation with system prompt
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]

            # Add context from previous conversation if available
            if context:
                # Parse context into messages (simplified)
                context_lines = context.strip().split('\n')
                for line in context_lines:
                    if line.startswith('Human: '):
                        messages.append({"role": "user", "content": line[7:]})
                    elif line.startswith('Assistant: '):
                        messages.append({"role": "assistant", "content": line[11:]})

            # Add current user prompt
            messages.append({"role": "user", "content": prompt})

            # Format for different model types
            if "mistral" in self.model_name.lower():
                formatted_prompt = self._format_mistral_chat(messages)
            elif "dialogpt" in self.model_name.lower():
                # DialoGPT works better with conversational format
                formatted_prompt = self._format_dialogpt_chat(messages)
            else:
                # Fallback formatting
                formatted_prompt = self._format_generic_chat(messages)

            # Generate response
            outputs = self.generator(
                formatted_prompt,
                max_length=len(self.tokenizer.encode(formatted_prompt)) + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )

            response = outputs[0]['generated_text']

            # Extract just the new response (after the input)
            response = self._extract_response(response, formatted_prompt)

            # Clean up response
            response = response.replace("<|endoftext|>", "").replace("</s>", "").strip()

            # If extraction returned an empty string, fall back to a safe template
            if not response:
                logger.warning("Model returned no generated text after extraction; using fallback response")
                return self._fallback_response(prompt), 0.5

            # Ensure reasonable length
            if len(response) > 1000:
                response = response[:1000] + "..."

            confidence = 0.88  # Higher confidence for instruct-tuned models

            return response, confidence

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return self._fallback_response(prompt), 0.5

    def _format_mistral_chat(self, messages: list) -> str:
        """Format messages using Mistral chat template"""
        formatted = "<s>"
        for msg in messages:
            if msg["role"] == "system":
                formatted += f"[INST] {msg['content']} [/INST]"
            elif msg["role"] == "user":
                formatted += f"[INST] {msg['content']} [/INST]"
            elif msg["role"] == "assistant":
                formatted += f" {msg['content']}</s><s>"
        return formatted

    def _format_dialogpt_chat(self, messages: list) -> str:
        """Format messages for DialoGPT conversational model"""
        # DialoGPT works best with conversational history
        # We'll include system prompt as context and format as conversation

        system_content = ""
        conversation_parts = []

        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                conversation_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                conversation_parts.append(f"Assistant: {msg['content']}")

        # Combine system context with conversation
        if system_content:
            formatted = f"System: {system_content}\n" + "\n".join(conversation_parts)
        else:
            formatted = "\n".join(conversation_parts)

        # Add the assistant prompt
        formatted += "\nAssistant:"

        return formatted

    def _format_generic_chat(self, messages: list) -> str:
        """Generic chat formatting fallback"""
        formatted = ""
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted += f"{role}: {content}\n"
        formatted += "Assistant:"
        return formatted

    def _extract_response(self, full_output: str, input_prompt: str) -> str:
        """Extract just the generated response from the full output"""
        # Remove the input prompt from the output
        if input_prompt in full_output:
            response = full_output.replace(input_prompt, "").strip()
        else:
            response = full_output.strip()

        # For Mistral, look for the response after [/INST]
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()

        # For DialoGPT, extract after "Assistant:"
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        # Clean up common artifacts
        response = response.split("<|endoftext|>")[0]
        response = response.split("</s>")[0]
        response = response.split("[INST]")[0]
        response = response.split("User:")[0]  # Don't include next user message

        return response.strip()

    def _fallback_response(self, prompt: str) -> str:
        """Enhanced fallback responses when model isn't available"""
        prompt_lower = prompt.lower()

        # More sophisticated pattern matching
        if any(word in prompt_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return "Hello! I'm a quantum-enhanced AI assistant. I'm here to help you with questions, discussions, and exploring ideas. What would you like to talk about?"

        elif any(word in prompt_lower for word in ['how are you', 'how do you do']):
            return "I'm functioning optimally! As a quantum-enhanced AI, I'm always ready to assist with complex questions, creative discussions, or technical problems. How can I help you today?"

        elif any(word in prompt_lower for word in ['what can you do', 'help', 'capabilities']):
            return """I can help you with:

🧠 **Complex Discussions**: Philosophy, science, technology, creativity
📊 **Problem Solving**: Technical challenges, logical puzzles, analysis
💭 **Creative Exploration**: Ideas, concepts, hypothetical scenarios
🔍 **Research & Learning**: Explaining concepts, exploring topics deeply
⚛️ **Quantum Concepts**: Quantum computing, physics, advanced mathematics

What interests you most? I'd love to dive deep into any of these areas!"""

        elif any(word in prompt_lower for word in ['quantum', 'physics', 'science']):
            return "Quantum physics is fascinating! The quantum world operates by different rules than our everyday experience - superposition, entanglement, uncertainty. Which aspect interests you? I can explain concepts, discuss implications, or explore current research."

        elif any(word in prompt_lower for word in ['ai', 'artificial intelligence', 'machine learning']):
            return "AI is a rapidly evolving field! From neural networks to quantum-enhanced algorithms, we're seeing incredible breakthroughs. I'm particularly interested in how quantum computing might revolutionize AI capabilities. What aspect would you like to explore?"

        elif any(word in prompt_lower for word in ['code', 'programming', 'python']):
            return "Programming is like giving instructions to a very logical and precise friend! Whether it's Python, algorithms, or software design, I can help debug, explain concepts, or explore new approaches. What are you working on?"

        else:
            # More engaging default response
            return f"That's an interesting point about '{prompt[:50]}...'. I'd love to explore this further with you. Could you tell me more about what you're thinking, or would you like me to share some related insights?"

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.model:
            return {"status": "no_model_loaded"}

        return {
            "model_name": self.model_name,
            "device": self.device,
            "model_size_mb": self._get_model_size_mb(),
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else 0,
            "max_position_embeddings": getattr(self.model.config, 'max_position_embeddings', 0)
        }

    def _get_model_size_mb(self) -> float:
        """Get model size in MB"""
        if not self.model:
            return 0.0

        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024 / 1024
        return round(size_mb, 2)

    # Multimodal capabilities
    def generate_multimodal_response(self, 
                                   prompt: str, 
                                   include_image: bool = None,
                                   context: str = "",
                                   image_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate both text and optionally image responses
        
        Args:
            prompt: User's input prompt
            include_image: Whether to generate an image (auto-detected if None)
            context: Previous conversation context
            image_params: Parameters for image generation
            
        Returns:
            Dict containing 'text', 'confidence', and optionally 'images' and 'image_paths'
        """
        
        # Generate text response
        text_response, confidence = self.generate_response(prompt, context)
        
        result = {
            "text": text_response,
            "confidence": confidence,
            "images": [],
            "image_paths": []
        }
        
        # Auto-detect if image generation is requested
        if include_image is None:
            image_keywords = [
                "generate image", "create image", "draw", "visualize", "show me",
                "nano banana", "picture of", "image of", "make an image"
            ]
            include_image = any(keyword in prompt.lower() for keyword in image_keywords)
        
        # Generate image if requested and available
        if include_image and self.enable_image_generation:
            try:
                logger.info("🎨 Generating image for prompt...")
                
                # Default image parameters
                default_params = {
                    "num_images": 1,
                    "width": 512,
                    "height": 512,
                    "enhancement_style": "enhance"
                }
                
                if image_params:
                    default_params.update(image_params)
                
                # Generate images
                images = self.image_generator.generate_image(prompt, **default_params)
                saved_paths = self.image_generator.save_images(images)
                
                result["images"] = images
                result["image_paths"] = saved_paths
                
                # Add image info to text response
                result["text"] += f"\n\n🎨 I've also generated {len(images)} image(s) based on your request! Check: {', '.join(saved_paths)}"
                
            except Exception as e:
                logger.error(f"Image generation failed: {e}")
                result["text"] += f"\n\n⚠️ I tried to generate an image but encountered an error: {str(e)}"
        
        elif include_image and not self.enable_image_generation:
            result["text"] += "\n\n📝 Note: Image generation is not available. Install diffusers library for visual capabilities!"
        
        return result

    def generate_image_only(self, prompt: str, **kwargs) -> List[str]:
        """Generate only images from a text prompt"""
        if not self.enable_image_generation:
            raise RuntimeError("Image generation not available")
        
        images = self.image_generator.generate_image(prompt, **kwargs)
        return self.image_generator.save_images(images)

    def is_image_generation_available(self) -> bool:
        """Check if image generation is available"""
        return self.enable_image_generation

    def get_capabilities(self) -> Dict[str, bool]:
        """Get information about available capabilities"""
        return {
            "text_generation": self.model is not None or self.encoded_backend is not None,
            "image_generation": self.enable_image_generation,
            "multimodal": self.enable_image_generation,
            "quantum_enhanced": self.use_encoded or self.enable_image_generation
        }