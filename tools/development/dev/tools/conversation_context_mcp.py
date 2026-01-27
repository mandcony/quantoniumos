# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Conversation Context Parser - MCP Integration
Parses conversation history and maintains context for intelligent responses
"""

import json
import time
from typing import List, Dict, Any
from datetime import datetime

class ConversationContextMCP:
    """Model Context Protocol for parsing and maintaining conversation state"""
    
    def __init__(self):
        self.conversation_history = []
        self.context_memory = {}
        self.topic_tracking = set()
        self.user_preferences = {}
        self.session_start = time.time()
    
    def parse_conversation_context(self, user_input: str, previous_messages: List[Dict] = None) -> Dict[str, Any]:
        """Parse conversation context for intelligent response generation"""
        
        # Add to conversation history
        message_entry = {
            'timestamp': time.time(),
            'user_input': user_input,
            'message_length': len(user_input),
            'question_count': user_input.count('?'),
            'topics_mentioned': self._extract_topics(user_input)
        }
        
        self.conversation_history.append(message_entry)
        
        # Build context for response
        context = {
            'current_input': user_input,
            'conversation_length': len(self.conversation_history),
            'recent_topics': list(self.topic_tracking)[-5:],  # Last 5 topics
            'session_duration': time.time() - self.session_start,
            'user_engagement_level': self._assess_engagement(user_input),
            'response_complexity_needed': self._determine_complexity(user_input),
            'conversation_flow': self._analyze_conversation_flow(),
            'context_continuity': self._check_context_continuity(user_input)
        }
        
        return context
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from user input"""
        text_lower = text.lower()
        topics = []
        
        # Science/Space topics
        if any(word in text_lower for word in ['space', 'planet', 'star', 'solar', 'universe', 'physics', 'quantum']):
            topics.append('science')
        
        # Math topics
        if any(word in text_lower for word in ['math', 'calculate', 'equation', 'number', 'fibonacci']):
            topics.append('mathematics')
        
        # Programming topics
        if any(word in text_lower for word in ['code', 'python', 'program', 'function', 'algorithm']):
            topics.append('programming')
        
        # Creative topics
        if any(word in text_lower for word in ['write', 'create', 'story', 'poem', 'art', 'design']):
            topics.append('creative')
        
        # Philosophy topics
        if any(word in text_lower for word in ['why', 'meaning', 'think', 'believe', 'philosophy']):
            topics.append('philosophy')
        
        # Update topic tracking
        for topic in topics:
            self.topic_tracking.add(topic)
        
        return topics
    
    def _assess_engagement(self, text: str) -> str:
        """Assess user engagement level"""
        if len(text) > 100:
            return 'high'
        elif len(text) > 20:
            return 'medium'
        else:
            return 'low'
    
    def _determine_complexity(self, text: str) -> str:
        """Determine how complex the response should be"""
        if '?' in text and len(text) > 50:
            return 'detailed'
        elif any(word in text.lower() for word in ['explain', 'how', 'why', 'what']):
            return 'comprehensive'
        elif len(text.split()) < 5:
            return 'engaging'
        else:
            return 'balanced'
    
    def _analyze_conversation_flow(self) -> str:
        """Analyze the flow of conversation"""
        if len(self.conversation_history) < 2:
            return 'opening'
        elif len(self.conversation_history) < 5:
            return 'developing'
        else:
            return 'established'
    
    def _check_context_continuity(self, current_input: str) -> bool:
        """Check if current input continues previous context"""
        if len(self.conversation_history) < 2:
            return False
        
        last_topics = self.conversation_history[-2]['topics_mentioned']
        current_topics = self._extract_topics(current_input)
        
        return bool(set(last_topics) & set(current_topics))
    
    def format_response_for_chatbox(self, response: str, max_length: int = 800) -> List[str]:
        """Format long responses for proper chatbox display"""
        if len(response) <= max_length:
            return [response]
        
        # Split into chunks at natural break points
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first
        paragraphs = response.split('\n\n')
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= max_length:
                if current_chunk:
                    current_chunk += '\n\n' + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def get_context_summary(self) -> str:
        """Get a summary of conversation context"""
        if not self.conversation_history:
            return "New conversation starting"
        
        topics = list(self.topic_tracking)
        duration = time.time() - self.session_start
        
        return f"Session: {len(self.conversation_history)} messages, {len(topics)} topics ({', '.join(topics)}), {duration/60:.1f}min"