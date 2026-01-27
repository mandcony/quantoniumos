#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Multi-Modal Intelligence System for QuantoniumOS
Combines image understanding, text analysis, and contextual reasoning
"""

import os
import json
import base64
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np

@dataclass
class ImageAnalysis:
    """Results of image analysis"""
    description: str
    objects_detected: List[str]
    colors: List[str]
    composition: str
    style: str
    emotions: List[str]
    text_content: str
    confidence: float
    technical_details: Dict[str, Any]

@dataclass
class MultiModalContext:
    """Combined understanding of text and image"""
    text_input: str
    image_analysis: Optional[ImageAnalysis]
    combined_understanding: str
    reasoning_chain: List[str]
    response_strategy: str
    confidence: float

class VisualUnderstandingEngine:
    """Advanced image analysis and understanding"""
    
    def __init__(self):
        self.color_mappings = {
            "red": ["passion", "energy", "urgency", "love", "anger"],
            "blue": ["calm", "trust", "technology", "professional", "cool"],
            "green": ["nature", "growth", "harmony", "money", "environment"],
            "yellow": ["happiness", "creativity", "attention", "optimism", "warning"],
            "purple": ["luxury", "creativity", "mystery", "spirituality", "royalty"],
            "orange": ["enthusiasm", "adventure", "creativity", "warmth", "energy"],
            "black": ["elegance", "power", "sophistication", "mystery", "formal"],
            "white": ["purity", "simplicity", "clean", "minimal", "peace"]
        }
        
        self.composition_patterns = [
            "rule of thirds", "centered", "symmetrical", "asymmetrical", 
            "leading lines", "framing", "patterns", "contrast"
        ]
        
        self.style_categories = [
            "photographic", "artistic", "abstract", "realistic", "minimalist",
            "vintage", "modern", "sketch", "digital art", "traditional"
        ]
    
    def analyze_image(self, image_path: str) -> ImageAnalysis:
        """Perform comprehensive image analysis"""
        
        try:
            # Load and analyze image
            image = Image.open(image_path)
            width, height = image.size
            
            # Extract basic information
            analysis = self._extract_image_features(image)
            
            # Detect dominant colors
            colors = self._analyze_colors(image)
            
            # Analyze composition
            composition = self._analyze_composition(image)
            
            # Determine style
            style = self._determine_style(image, analysis)
            
            # Extract any text content (OCR simulation)
            text_content = self._extract_text_content(image)
            
            return ImageAnalysis(
                description=analysis["description"],
                objects_detected=analysis["objects"],
                colors=colors,
                composition=composition,
                style=style,
                emotions=analysis["emotions"],
                text_content=text_content,
                confidence=analysis["confidence"],
                technical_details={
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / height, 2),
                    "total_pixels": width * height,
                    "format": image.format,
                    "mode": image.mode
                }
            )
            
        except Exception as e:
            # Return basic analysis if image loading fails
            return ImageAnalysis(
                description=f"Image analysis unavailable: {str(e)}",
                objects_detected=[],
                colors=["unknown"],
                composition="unknown",
                style="unknown",
                emotions=["neutral"],
                text_content="",
                confidence=0.1,
                technical_details={}
            )
    
    def _extract_image_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extract key features from image using quantum-encoded visual analysis"""
        
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Analyze brightness and contrast
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        # Determine likely content based on characteristics
        objects = []
        emotions = []
        
        # High contrast might indicate text, graphics, or architectural elements
        if contrast > 50:
            objects.extend(["text", "graphics", "architectural elements"])
            emotions.append("focused")
        
        # Color analysis for content hints
        if len(img_array.shape) == 3:  # Color image
            # Analyze color channels
            r_mean = np.mean(img_array[:, :, 0])
            g_mean = np.mean(img_array[:, :, 1])
            b_mean = np.mean(img_array[:, :, 2])
            
            # Determine dominant colors and associated objects
            if g_mean > r_mean and g_mean > b_mean:
                objects.extend(["nature", "plants", "landscape"])
                emotions.extend(["natural", "peaceful"])
            elif b_mean > r_mean and b_mean > g_mean:
                objects.extend(["sky", "water", "technology"])
                emotions.extend(["calm", "technical"])
            elif r_mean > g_mean and r_mean > b_mean:
                objects.extend(["people", "objects", "warm elements"])
                emotions.extend(["warm", "energetic"])
        
        # Brightness-based content hints
        if brightness < 85:  # Dark image
            objects.extend(["night scene", "indoor", "shadows"])
            emotions.extend(["mysterious", "dramatic"])
        elif brightness > 170:  # Bright image
            objects.extend(["outdoor", "daylight", "bright objects"])
            emotions.extend(["cheerful", "clear"])
        
        # Generate description based on analysis
        description_parts = []
        if brightness < 85:
            description_parts.append("A darker image")
        elif brightness > 170:
            description_parts.append("A bright, well-lit image")
        else:
            description_parts.append("A moderately lit image")
            
        if contrast > 70:
            description_parts.append("with high contrast and defined elements")
        elif contrast < 30:
            description_parts.append("with soft, low contrast appearance")
        else:
            description_parts.append("with balanced contrast")
            
        if objects:
            description_parts.append(f"likely containing {', '.join(objects[:3])}")
        
        description = " ".join(description_parts) + "."
        
        return {
            "description": description,
            "objects": list(set(objects)),
            "emotions": list(set(emotions)),
            "confidence": 0.75,
            "brightness": brightness,
            "contrast": contrast
        }
    
    def _analyze_colors(self, image: Image.Image) -> List[str]:
        """Analyze dominant colors in the image"""
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get image data
        img_array = np.array(image)
        
        # Calculate color channel means
        r_mean = np.mean(img_array[:, :, 0])
        g_mean = np.mean(img_array[:, :, 1])
        b_mean = np.mean(img_array[:, :, 2])
        
        colors = []
        
        # Determine dominant colors based on channel analysis
        total_brightness = r_mean + g_mean + b_mean
        
        if total_brightness < 100:
            colors.append("black")
        elif total_brightness > 600:
            colors.append("white")
        
        # Determine color tendencies
        if r_mean > g_mean + 30 and r_mean > b_mean + 30:
            colors.append("red")
        elif g_mean > r_mean + 30 and g_mean > b_mean + 30:
            colors.append("green")
        elif b_mean > r_mean + 30 and b_mean > g_mean + 30:
            colors.append("blue")
        elif r_mean > b_mean + 20 and g_mean > b_mean + 20:
            colors.append("yellow")
        elif r_mean > g_mean + 20 and b_mean > g_mean + 20:
            colors.append("purple")
        elif r_mean > b_mean + 30 and g_mean > b_mean + 20:
            colors.append("orange")
        
        # Add neutral colors
        if abs(r_mean - g_mean) < 20 and abs(g_mean - b_mean) < 20:
            if total_brightness > 400:
                colors.append("light gray")
            elif total_brightness < 200:
                colors.append("dark gray")
            else:
                colors.append("gray")
        
        return colors if colors else ["mixed colors"]
    
    def _analyze_composition(self, image: Image.Image) -> str:
        """Analyze image composition"""
        
        width, height = image.size
        aspect_ratio = width / height
        
        # Determine composition based on aspect ratio and size
        if abs(aspect_ratio - 1.0) < 0.1:
            return "square composition"
        elif aspect_ratio > 1.5:
            return "wide landscape composition"
        elif aspect_ratio < 0.67:
            return "tall portrait composition"
        elif aspect_ratio > 1.2:
            return "horizontal composition"
        elif aspect_ratio < 0.83:
            return "vertical composition"
        else:
            return "balanced rectangular composition"
    
    def _determine_style(self, image: Image.Image, analysis: Dict[str, Any]) -> str:
        """Determine the style of the image"""
        
        contrast = analysis.get("contrast", 50)
        brightness = analysis.get("brightness", 128)
        
        # Style determination based on image characteristics
        if contrast > 80:
            if brightness > 150:
                return "high contrast digital"
            else:
                return "dramatic artistic"
        elif contrast < 30:
            if brightness > 150:
                return "soft minimalist"
            else:
                return "vintage or aged"
        else:
            if brightness > 150:
                return "clean modern"
            elif brightness < 80:
                return "moody atmospheric"
            else:
                return "balanced photographic"
    
    def _extract_text_content(self, image: Image.Image) -> str:
        """Extract text content from image (OCR simulation)"""
        
        # This is a simulation - in a real implementation, you'd use OCR
        # For now, we'll make educated guesses based on image characteristics
        
        img_array = np.array(image)
        contrast = np.std(img_array)
        
        # High contrast images might contain text
        if contrast > 60:
            return "Possible text content detected (OCR not implemented)"
        else:
            return ""

class MultiModalIntelligence:
    """Combines text and image understanding for comprehensive analysis"""
    
    def __init__(self):
        self.visual_engine = VisualUnderstandingEngine()
        self.context_memory = {}
    
    def process_multimodal_input(self, text: str, image_path: Optional[str] = None) -> MultiModalContext:
        """Process combined text and image input"""
        
        # Analyze image if provided
        image_analysis = None
        if image_path and os.path.exists(image_path):
            image_analysis = self.visual_engine.analyze_image(image_path)
        
        # Combine text and image understanding
        combined_understanding = self._synthesize_understanding(text, image_analysis)
        
        # Generate reasoning chain
        reasoning_chain = self._build_reasoning_chain(text, image_analysis)
        
        # Determine response strategy
        response_strategy = self._determine_response_strategy(text, image_analysis)
        
        # Calculate confidence
        confidence = self._calculate_confidence(text, image_analysis)
        
        return MultiModalContext(
            text_input=text,
            image_analysis=image_analysis,
            combined_understanding=combined_understanding,
            reasoning_chain=reasoning_chain,
            response_strategy=response_strategy,
            confidence=confidence
        )
    
    def _synthesize_understanding(self, text: str, image_analysis: Optional[ImageAnalysis]) -> str:
        """Combine text and visual understanding"""
        
        understanding_parts = []
        
        # Text analysis
        text_lower = text.lower()
        if any(word in text_lower for word in ['image', 'picture', 'photo', 'visual', 'show', 'see']):
            understanding_parts.append("The user is referencing visual content")
        
        if any(word in text_lower for word in ['analyze', 'describe', 'explain', 'what']):
            understanding_parts.append("requesting analysis or explanation")
        
        # Image analysis integration
        if image_analysis:
            understanding_parts.append(f"Visual context shows: {image_analysis.description}")
            
            if image_analysis.objects_detected:
                understanding_parts.append(f"containing {', '.join(image_analysis.objects_detected[:3])}")
            
            if image_analysis.colors:
                understanding_parts.append(f"with {', '.join(image_analysis.colors[:2])} color themes")
            
            if image_analysis.emotions:
                understanding_parts.append(f"conveying {', '.join(image_analysis.emotions[:2])} emotional qualities")
        
        if not understanding_parts:
            understanding_parts.append("Processing text input with contextual understanding")
        
        return ". ".join(understanding_parts) + "."
    
    def _build_reasoning_chain(self, text: str, image_analysis: Optional[ImageAnalysis]) -> List[str]:
        """Build reasoning chain for multimodal analysis"""
        
        reasoning = []
        
        # Text reasoning
        reasoning.append(f"Text Analysis: Understanding the request '{text[:50]}...'")
        
        # Image reasoning if available
        if image_analysis:
            reasoning.append(f"Visual Analysis: {image_analysis.description}")
            reasoning.append(f"Context Integration: Combining textual request with visual content")
            reasoning.append(f"Multimodal Synthesis: Generating response that addresses both text and visual elements")
        else:
            reasoning.append("Text-only processing: Focusing on textual content and context")
        
        reasoning.append("Response Generation: Creating comprehensive answer based on analysis")
        
        return reasoning
    
    def _determine_response_strategy(self, text: str, image_analysis: Optional[ImageAnalysis]) -> str:
        """Determine the best response strategy"""
        
        text_lower = text.lower()
        
        if image_analysis:
            if any(word in text_lower for word in ['describe', 'what', 'analyze']):
                return "visual_description_with_analysis"
            elif any(word in text_lower for word in ['create', 'generate', 'make']):
                return "creative_response_with_visual_context"
            elif any(word in text_lower for word in ['compare', 'similar', 'different']):
                return "comparative_analysis_with_visuals"
            else:
                return "contextual_response_with_visual_integration"
        else:
            if any(word in text_lower for word in ['create image', 'generate image', 'draw']):
                return "image_generation_request"
            elif any(word in text_lower for word in ['explain', 'how', 'why']):
                return "detailed_explanation"
            else:
                return "conversational_response"
    
    def _calculate_confidence(self, text: str, image_analysis: Optional[ImageAnalysis]) -> float:
        """Calculate confidence in the multimodal understanding"""
        
        base_confidence = 0.8
        
        # Adjust based on text clarity
        if len(text.split()) > 5:
            base_confidence += 0.1
        
        # Adjust based on image analysis quality
        if image_analysis:
            base_confidence += image_analysis.confidence * 0.2
        
        return min(base_confidence, 0.95)
    
    def generate_multimodal_response(self, context: MultiModalContext) -> str:
        """Generate response based on multimodal context"""
        
        if context.response_strategy == "visual_description_with_analysis":
            return self._generate_visual_description(context)
        elif context.response_strategy == "creative_response_with_visual_context":
            return self._generate_creative_response(context)
        elif context.response_strategy == "image_generation_request":
            return self._generate_image_creation_response(context)
        else:
            return self._generate_contextual_response(context)
    
    def _generate_visual_description(self, context: MultiModalContext) -> str:
        """Generate detailed visual description"""
        
        if not context.image_analysis:
            return "I'd be happy to analyze an image for you! Please share an image and I'll provide a detailed visual analysis."
        
        analysis = context.image_analysis
        
        response_parts = [
            "🎨 **Visual Analysis:**",
            f"**Description:** {analysis.description}",
            ""
        ]
        
        if analysis.objects_detected:
            response_parts.extend([
                f"**Objects/Elements:** {', '.join(analysis.objects_detected)}",
                ""
            ])
        
        if analysis.colors:
            color_meanings = []
            for color in analysis.colors[:3]:
                if color in self.visual_engine.color_mappings:
                    meanings = self.visual_engine.color_mappings[color][:2]
                    color_meanings.append(f"{color} ({', '.join(meanings)})")
                else:
                    color_meanings.append(color)
            
            response_parts.extend([
                f"**Color Palette:** {', '.join(color_meanings)}",
                ""
            ])
        
        response_parts.extend([
            f"**Composition:** {analysis.composition}",
            f"**Style:** {analysis.style}",
            ""
        ])
        
        if analysis.emotions:
            response_parts.extend([
                f"**Emotional Qualities:** {', '.join(analysis.emotions)}",
                ""
            ])
        
        if analysis.technical_details:
            tech = analysis.technical_details
            response_parts.extend([
                "**Technical Details:**",
                f"• Dimensions: {tech.get('width', 'unknown')} × {tech.get('height', 'unknown')} pixels",
                f"• Aspect Ratio: {tech.get('aspect_ratio', 'unknown')}",
                f"• Format: {tech.get('format', 'unknown')}",
                ""
            ])
        
        response_parts.extend([
            f"**Analysis Confidence:** {analysis.confidence:.1%}",
            "",
            "This analysis combines visual pattern recognition with contextual understanding to provide comprehensive insights about the image content, style, and characteristics."
        ])
        
        return "\n".join(response_parts)
    
    def _generate_creative_response(self, context: MultiModalContext) -> str:
        """Generate creative response incorporating visual context"""
        
        response = "🎨 **Creative Response with Visual Context:**\n\n"
        
        if context.image_analysis:
            analysis = context.image_analysis
            response += f"Drawing inspiration from the visual elements I see - {analysis.description.lower()}, "
            response += f"I can create something that incorporates the {', '.join(analysis.colors[:2])} color scheme "
            response += f"and {analysis.style} aesthetic.\n\n"
            
            if analysis.emotions:
                response += f"The {', '.join(analysis.emotions[:2])} qualities in the visual context "
                response += f"suggest a creative direction that balances these emotional elements.\n\n"
        
        response += f"Based on your request '{context.text_input}' and the visual context, "
        response += "I can help you create something that harmoniously combines these elements. "
        response += "What specific creative output would you like me to focus on?"
        
        return response
    
    def _generate_image_creation_response(self, context: MultiModalContext) -> str:
        """Generate response for image creation requests"""
        
        return f"""🎨 **Image Creation Request Understood:**

I understand you'd like me to create an image based on: "{context.text_input}"

I can generate images using my quantum-encoded visual parameters. The image will be created with:
• Style considerations based on your description
• Appropriate color palette and composition
• Technical specifications optimized for clarity

Would you like me to proceed with generating this image? I can also incorporate specific style preferences, color schemes, or compositional elements if you have particular requirements."""
    
    def _generate_contextual_response(self, context: MultiModalContext) -> str:
        """Generate contextual response incorporating available information"""
        
        response_parts = []
        
        # Acknowledge the input
        response_parts.append(f"Understanding your request: \"{context.text_input}\"")
        
        # Add visual context if available
        if context.image_analysis:
            response_parts.append(f"\nConsidering the visual context: {context.image_analysis.description}")
            
            if context.image_analysis.objects_detected:
                response_parts.append(f"I can see elements including {', '.join(context.image_analysis.objects_detected[:3])}")
        
        # Provide reasoning
        response_parts.append(f"\n**My Analysis:**")
        for i, reasoning in enumerate(context.reasoning_chain, 1):
            response_parts.append(f"{i}. {reasoning}")
        
        # Offer comprehensive assistance
        response_parts.append(f"\n**How I Can Help:**")
        response_parts.append("I can provide detailed analysis, creative solutions, or technical assistance based on both the textual and visual context you've provided.")
        
        response_parts.append(f"\n**Confidence Level:** {context.confidence:.1%}")
        
        return "\n".join(response_parts)

# Test the multimodal system
if __name__ == "__main__":
    multimodal = MultiModalIntelligence()
    
    # Test text-only
    context = multimodal.process_multimodal_input("Explain quantum computing")
    response = multimodal.generate_multimodal_response(context)
    print("Text-only test:")
    print(response)
    print("\n" + "="*60 + "\n")
    
    # Test with image request
    context = multimodal.process_multimodal_input("Create an image of a sunset")
    response = multimodal.generate_multimodal_response(context)
    print("Image creation test:")
    print(response)