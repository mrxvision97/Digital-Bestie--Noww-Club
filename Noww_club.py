import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import add_messages
import os
import uuid
import json
import time
import asyncio
from typing import TypedDict, Annotated, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain_core.runnables import RunnablePassthrough
from pathlib import Path
import re
import spacy
from dateutil import parser
import pytz
from difflib import get_close_matches

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If model not found, download it
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

class FlowManager:
    def __init__(self):
        self.active_flow = None
        self.flow_step = 0
        self.flow_data = {}
        self.flow_state = {}
        self.paused_flows = []
        self.last_flow_context = None
        self.last_answered_step = 0  # Track the last step that was answered

    def start_flow(self, flow_type):
        """Start a new flow"""
        self.active_flow = flow_type
        self.flow_step = 1
        self.flow_data = {}
        self.flow_state = {}
        self.last_answered_step = 0
        print(f"Started new flow: {flow_type}")

    def pause_flow(self):
        """Pause the current flow"""
        if self.active_flow:
            # Store complete flow context
            flow_context = {
                'type': self.active_flow,
                'step': self.flow_step,
                'data': self.flow_data.copy(),
                'state': self.flow_state.copy(),
                'last_answered_step': self.last_answered_step
            }
            self.paused_flows.append(flow_context)
            self.last_flow_context = flow_context
            
            # Clear current flow state
            self.active_flow = None
            self.flow_step = 0
            self.flow_data = {}
            self.flow_state = {}
            self.last_answered_step = 0
            print(f"Paused flow: {flow_context['type']}")

    def resume_last_flow(self):
        """Resume the last paused flow"""
        if self.paused_flows:
            last_flow = self.paused_flows.pop()
            self.active_flow = last_flow['type']
            self.flow_step = last_flow['step']
            self.flow_data = last_flow['data'].copy()
            self.flow_state = last_flow['state'].copy()
            self.last_answered_step = last_flow.get('last_answered_step', 0)
            self.last_flow_context = last_flow.copy()
            
            # Get the appropriate resume message based on flow type and step
            resume_message = self._get_resume_message(last_flow['type'], last_flow['step'])
            print(f"Resumed flow: {self.active_flow} at step {self.flow_step}")
            return True, resume_message
        return False, None

    def _get_resume_message(self, flow_type, step):
        """Get appropriate resume message based on flow type and step"""
        if flow_type == "habit":
            if step == 1:
                return "What habit would you like to build?"
            elif step == 2:
                return "How often would you like to practice this habit? (e.g., daily, weekly, specific days)"
            elif step == 3:
                return "What's your motivation for building this habit? This will help you stay committed."
            elif step == 4:
                return "How would you like to receive reminders? (Email, SMS, or Push Notification)"
            elif step == 5:
                return "What time would you like to receive the reminder? (e.g., 9:00 AM)"
        elif flow_type == "goal":
            if step == 1:
                return "What goal would you like to achieve?"
            elif step == 2:
                return "When would you like to achieve this goal by?"
            elif step == 3:
                return "What are the steps you'll take to achieve this goal?"
            elif step == 4:
                return "How would you like to receive reminders? (Email, SMS, or Push Notification)"
            elif step == 5:
                return "What time would you like to receive the reminder? (e.g., 9:00 AM)"
        elif flow_type == "reminder":
            if step == 1:
                return "What would you like to be reminded about?"
            elif step == 2:
                return "How would you like to receive reminders? (Email, SMS, or Push Notification)"
            elif step == 3:
                return "What time would you like to receive the reminder? (e.g., 9:00 AM)"
        return "Let's continue with your flow."

    def set_flow_data(self, key, value):
        """Set data for the current flow"""
        self.flow_data[key] = value
        self.last_answered_step = self.flow_step  # Update last answered step when data is set

    def get_flow_data(self, key=None):
        """Get data for the current flow"""
        if key:
            return self.flow_data.get(key)
        return self.flow_data

    def clear_flow_data(self):
        """Clear all flow data"""
        self.active_flow = None
        self.flow_step = 0
        self.flow_data = {}
        self.flow_state = {}
        self.last_answered_step = 0
        self.paused_flows = []
        self.last_flow_context = None

    def is_flow_active(self):
        """Check if there is an active flow"""
        return self.active_flow is not None

    def has_paused_flows(self):
        """Check if there are any paused flows"""
        return len(self.paused_flows) > 0

    def get_flow_state(self):
        """Get the current flow state"""
        return self.flow_state

    def set_flow_state(self, key, value):
        """Set a value in the flow state"""
        self.flow_state[key] = value

    def get_flow_state_value(self, key):
        """Get a value from the flow state"""
        return self.flow_state.get(key)

    def get_active_flow_type(self):
        """Get the type of the active flow"""
        return self.active_flow

    def get_flow_step(self):
        """Get the current flow step"""
        return self.flow_step

    def increment_flow_step(self):
        """Increment the flow step"""
        self.flow_step += 1
        return self.flow_step

    def set_flow_step(self, step):
        """Set the flow step"""
        self.flow_step = step
        return self.flow_step

class TimeExtractor:
    def __init__(self):
        self.time_patterns = {
            r'morning at (\d{1,2})': lambda x: f"{x}:00 AM",
            r'evening at (\d{1,2})': lambda x: f"{x}:00 PM",
            r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)': lambda h, m, p: f"{h}:{m or '00'} {p.upper()}",
            r'(\d{1,2}):(\d{2})': lambda h, m: f"{h}:{m}",
        }
        self.fuzzy_time_map = {
            "morning": "6:00 AM",
            "early morning": "5:00 AM",
            "late morning": "10:00 AM",
            "afternoon": "2:00 PM",
            "evening": "6:00 PM",
            "night": "9:00 PM",
            "late night": "11:00 PM"
        }

    def extract_time(self, text: str) -> Optional[str]:
        try:
            # Check for fuzzy time references
            for fuzzy_time, default_time in self.fuzzy_time_map.items():
                if fuzzy_time in text.lower():
                    return default_time

            # Check for specific time patterns
            for pattern, formatter in self.time_patterns.items():
                match = re.search(pattern, text.lower())
                if match:
                    return formatter(*match.groups())

            return None
        except Exception as e:
            print(f"Error in time extraction: {str(e)}")
            return None

    def clarify_ambiguous_time(self, time_str: str) -> str:
        """Handle ambiguous time inputs."""
        if "morning" in time_str.lower() and "at" in time_str.lower():
            return "Just to confirm — did you mean 6:00 AM or 7:00 AM?"
        return time_str

class IntentDetector:
    def __init__(self):
        self.intent_patterns = {
            'habit': [
                r'create\s+(?:a|an)?\s*habit',
                r'start\s+(?:a|an)?\s*habit',
                r'build\s+(?:a|an)?\s*habit',
                r'new\s+habit',
                r'want\s+to\s+(?:start|create|build)\s+(?:a|an)?\s*habit',
                r'daily\s+habit',
                r'weekly\s+habit',
                r'regular\s+habit'
            ],
            'goal': [
                r'set\s+(?:a|an)?\s*goal',
                r'create\s+(?:a|an)?\s*goal',
                r'new\s+goal',
                r'want\s+to\s+(?:set|create)\s+(?:a|an)?\s*goal',
                r'target',
                r'objective'
            ],
            'reminder': [
                r'set\s+(?:a|an)?\s*reminder',
                r'create\s+(?:a|an)?\s*reminder',
                r'new\s+reminder',
                r'remind\s+me',
                r'set\s+reminder',
                r'create\s+reminder'
            ],
            'emotional_support': [
                r'(?:i|i\'m|i am)\s+(?:feeling|feel)\s+(?:sad|happy|angry|anxious|worried|stressed|depressed|lonely|tired|exhausted|overwhelmed|frustrated|disappointed|hurt|scared|nervous|confused|lost|empty|numb|hopeless|helpless|worthless|guilty|ashamed|embarrassed|jealous|envious|grateful|blessed|lucky|fortunate|excited|thrilled|joyful|peaceful|calm|relaxed|content|satisfied|proud|confident|strong|powerful|capable|worthy|loved|appreciated|valued|understood|heard|seen|supported|encouraged|motivated|inspired|hopeful|optimistic|positive|negative|neutral|okay|fine|alright|good|great|amazing|wonderful|fantastic|terrible|horrible|awful|bad|not good|not great|not okay|not fine|not alright)',
                r'(?:i|i\'m|i am)\s+(?:sad|happy|angry|anxious|worried|stressed|depressed|lonely|tired|exhausted|overwhelmed|frustrated|disappointed|hurt|scared|nervous|confused|lost|empty|numb|hopeless|helpless|worthless|guilty|ashamed|embarrassed|jealous|envious|grateful|blessed|lucky|fortunate|excited|thrilled|joyful|peaceful|calm|relaxed|content|satisfied|proud|confident|strong|powerful|capable|worthy|loved|appreciated|valued|understood|heard|seen|supported|encouraged|motivated|inspired|hopeful|optimistic|positive|negative|neutral|okay|fine|alright|good|great|amazing|wonderful|fantastic|terrible|horrible|awful|bad|not good|not great|not okay|not fine|not alright)',
                r'(?:feeling|feel)\s+(?:sad|happy|angry|anxious|worried|stressed|depressed|lonely|tired|exhausted|overwhelmed|frustrated|disappointed|hurt|scared|nervous|confused|lost|empty|numb|hopeless|helpless|worthless|guilty|ashamed|embarrassed|jealous|envious|grateful|blessed|lucky|fortunate|excited|thrilled|joyful|peaceful|calm|relaxed|content|satisfied|proud|confident|strong|powerful|capable|worthy|loved|appreciated|valued|understood|heard|seen|supported|encouraged|motivated|inspired|hopeful|optimistic|positive|negative|neutral|okay|fine|alright|good|great|amazing|wonderful|fantastic|terrible|horrible|awful|bad|not good|not great|not okay|not fine|not alright)',
                r'(?:i|i\'m|i am)\s+(?:having|going through|experiencing|dealing with)\s+(?:a|an)\s+(?:hard|difficult|tough|rough|bad|good|great|amazing|wonderful|fantastic|terrible|horrible|awful)\s+(?:time|day|week|month|year|period|phase|moment|situation|circumstance|experience)',
                r'(?:i|i\'m|i am)\s+(?:struggling|having trouble|having difficulty|having a hard time|having a difficult time|having a tough time|having a rough time|having a bad time|having a good time|having a great time|having an amazing time|having a wonderful time|having a fantastic time|having a terrible time|having a horrible time|having an awful time)',
                r'(?:i|i\'m|i am)\s+(?:not|don\'t|do not)\s+(?:feeling|feel)\s+(?:well|good|great|amazing|wonderful|fantastic|terrible|horrible|awful|bad|not good|not great|not okay|not fine|not alright)',
                r'(?:i|i\'m|i am)\s+(?:not|don\'t|do not)\s+(?:doing|going)\s+(?:well|good|great|amazing|wonderful|fantastic|terrible|horrible|awful|bad|not good|not great|not okay|not fine|not alright)',
                r'(?:i|i\'m|i am)\s+(?:not|don\'t|do not)\s+(?:okay|fine|alright|good|great|amazing|wonderful|fantastic|terrible|horrible|awful|bad|not good|not great|not okay|not fine|not alright)',
                r'(?:i|i\'m|i am)\s+(?:not|don\'t|do not)\s+(?:know|understand|get|see|feel|think|believe|trust|hope|wish|want|need|care|matter|count|belong|fit|work|function|operate|perform|succeed|thrive|survive|exist|live|breathe|move|think|feel|act|be)',
                r'(?:i|i\'m|i am)\s+(?:lost|confused|stuck|trapped|alone|lonely|isolated|abandoned|rejected|unwanted|unloved|unappreciated|unvalued|misunderstood|unheard|unseen|unsupported|unencouraged|unmotivated|uninspired|hopeless|helpless|worthless|guilty|ashamed|embarrassed|jealous|envious|grateful|blessed|lucky|fortunate|excited|thrilled|joyful|peaceful|calm|relaxed|content|satisfied|proud|confident|strong|powerful|capable|worthy|loved|appreciated|valued|understood|heard|seen|supported|encouraged|motivated|inspired|hopeful|optimistic|positive|negative|neutral|okay|fine|alright|good|great|amazing|wonderful|fantastic|terrible|horrible|awful|bad|not good|not great|not okay|not fine|not alright)',
                r'(?:i|i\'m|i am)\s+(?:tired|exhausted|drained|depleted|empty|numb|dead|gone|done|finished|over|through|out|away|far|distant|separate|different|other|else|more|less|better|worse|same|similar|different|opposite|contrary|contrasting|conflicting|contradicting|contradictory|inconsistent|incompatible|incomprehensible|inconceivable|unimaginable|unthinkable|unbelievable|unreal|untrue|false|wrong|incorrect|inaccurate|imprecise|inexact|approximate|rough|general|vague|ambiguous|unclear|uncertain|unsure|doubtful|dubious|suspicious|skeptical|cynical|pessimistic|negative|neutral|positive|optimistic|hopeful|inspired|motivated|encouraged|supported|seen|heard|understood|valued|appreciated|loved|worthy|capable|powerful|strong|confident|proud|satisfied|content|relaxed|calm|peaceful|joyful|thrilled|excited|fortunate|lucky|blessed|grateful|envious|jealous|embarrassed|ashamed|guilty|worthless|helpless|hopeless|inspired|motivated|encouraged|supported|seen|heard|understood|valued|appreciated|loved|worthy|capable|powerful|strong|confident|proud|satisfied|content|relaxed|calm|peaceful|joyful|thrilled|excited|fortunate|lucky|blessed|grateful|envious|jealous|embarrassed|ashamed|guilty|worthless|helpless|hopeless)',
                r'(?:i|i\'m|i am)\s+(?:overwhelmed|frustrated|disappointed|hurt|scared|nervous|confused|lost|empty|numb|hopeless|helpless|worthless|guilty|ashamed|embarrassed|jealous|envious|grateful|blessed|lucky|fortunate|excited|thrilled|joyful|peaceful|calm|relaxed|content|satisfied|proud|confident|strong|powerful|capable|worthy|loved|appreciated|valued|understood|heard|seen|supported|encouraged|motivated|inspired|hopeful|optimistic|positive|negative|neutral|okay|fine|alright|good|great|amazing|wonderful|fantastic|terrible|horrible|awful|bad|not good|not great|not okay|not fine|not alright)',
                r'(?:i|i\'m|i am)\s+(?:having|going through|experiencing|dealing with)\s+(?:a|an)\s+(?:hard|difficult|tough|rough|bad|good|great|amazing|wonderful|fantastic|terrible|horrible|awful)\s+(?:time|day|week|month|year|period|phase|moment|situation|circumstance|experience)',
                r'(?:i|i\'m|i am)\s+(?:struggling|having trouble|having difficulty|having a hard time|having a difficult time|having a tough time|having a rough time|having a bad time|having a good time|having a great time|having an amazing time|having a wonderful time|having a fantastic time|having a terrible time|having a horrible time|having an awful time)',
                r'(?:i|i\'m|i am)\s+(?:not|don\'t|do not)\s+(?:feeling|feel)\s+(?:well|good|great|amazing|wonderful|fantastic|terrible|horrible|awful|bad|not good|not great|not okay|not fine|not alright)',
                r'(?:i|i\'m|i am)\s+(?:not|don\'t|do not)\s+(?:doing|going)\s+(?:well|good|great|amazing|wonderful|fantastic|terrible|horrible|awful|bad|not good|not great|not okay|not fine|not alright)',
                r'(?:i|i\'m|i am)\s+(?:not|don\'t|do not)\s+(?:okay|fine|alright|good|great|amazing|wonderful|fantastic|terrible|horrible|awful|bad|not good|not great|not okay|not fine|not alright)',
                r'(?:i|i\'m|i am)\s+(?:not|don\'t|do not)\s+(?:know|understand|get|see|feel|think|believe|trust|hope|wish|want|need|care|matter|count|belong|fit|work|function|operate|perform|succeed|thrive|survive|exist|live|breathe|move|think|feel|act|be)',
                r'(?:i|i\'m|i am)\s+(?:lost|confused|stuck|trapped|alone|lonely|isolated|abandoned|rejected|unwanted|unloved|unappreciated|unvalued|misunderstood|unheard|unseen|unsupported|unencouraged|unmotivated|uninspired|hopeless|helpless|worthless|guilty|ashamed|embarrassed|jealous|envious|grateful|blessed|lucky|fortunate|excited|thrilled|joyful|peaceful|calm|relaxed|content|satisfied|proud|confident|strong|powerful|capable|worthy|loved|appreciated|valued|understood|heard|seen|supported|encouraged|motivated|inspired|hopeful|optimistic|positive|negative|neutral|okay|fine|alright|good|great|amazing|wonderful|fantastic|terrible|horrible|awful|bad|not good|not great|not okay|not fine|not alright)',
                r'(?:i|i\'m|i am)\s+(?:tired|exhausted|drained|depleted|empty|numb|dead|gone|done|finished|over|through|out|away|far|distant|separate|different|other|else|more|less|better|worse|same|similar|different|opposite|contrary|contrasting|conflicting|contradicting|contradictory|inconsistent|incompatible|incomprehensible|inconceivable|unimaginable|unthinkable|unbelievable|unreal|untrue|false|wrong|incorrect|inaccurate|imprecise|inexact|approximate|rough|general|vague|ambiguous|unclear|uncertain|unsure|doubtful|dubious|suspicious|skeptical|cynical|pessimistic|negative|neutral|positive|optimistic|hopeful|inspired|motivated|encouraged|supported|seen|heard|understood|valued|appreciated|loved|worthy|capable|powerful|strong|confident|proud|satisfied|content|relaxed|calm|peaceful|joyful|thrilled|excited|fortunate|lucky|blessed|grateful|envious|jealous|embarrassed|ashamed|guilty|worthless|helpless|hopeless|inspired|motivated|encouraged|supported|seen|heard|understood|valued|appreciated|loved|worthy|capable|powerful|strong|confident|proud|satisfied|content|relaxed|calm|peaceful|joyful|thrilled|excited|fortunate|lucky|blessed|grateful|envious|jealous|embarrassed|ashamed|guilty|worthless|helpless|hopeless)',
                r'(?:i|i\'m|i am)\s+(?:overwhelmed|frustrated|disappointed|hurt|scared|nervous|confused|lost|empty|numb|hopeless|helpless|worthless|guilty|ashamed|embarrassed|jealous|envious|grateful|blessed|lucky|fortunate|excited|thrilled|joyful|peaceful|calm|relaxed|content|satisfied|proud|confident|strong|powerful|capable|worthy|loved|appreciated|valued|understood|heard|seen|supported|encouraged|motivated|inspired|hopeful|optimistic|positive|negative|neutral|okay|fine|alright|good|great|amazing|wonderful|fantastic|terrible|horrible|awful|bad|not good|not great|not okay|not fine|not alright)'
            ],
            'casual_chat': [
                r'how\s+are\s+you',
                r'what\'?s\s+up',
                r'hello',
                r'hi\s+there',
                r'hey',
                r'good\s+(?:morning|afternoon|evening)',
                r'btw',
                r'by\s+the\s+way',
                r'just\s+wondering',
                r'quick\s+question',
                r'can\s+i\s+ask',
                r'what\s+do\s+you\s+think',
                r'what\'?s\s+your\s+opinion',
                r'what\'?s\s+your\s+favorite',
                r'do\s+you\s+like',
                r'have\s+you\s+ever',
                r'what\'?s\s+the\s+weather',
                r'how\'?s\s+your\s+day',
                r'what\'?s\s+new',
                r'how\'?s\s+it\s+going'
            ]
        }

    def detect_intent(self, text: str) -> Tuple[Optional[str], float]:
        """Detect intent using NLP and pattern matching"""
        doc = nlp(text.lower())
        
        # Check for intent-related entities
        for ent in doc.ents:
            if ent.label_ in ["EVENT", "ACTIVITY"]:
                if "habit" in ent.text:
                    return "habit", 0.9
                elif "goal" in ent.text:
                    return "goal", 0.9
                elif "reminder" in ent.text:
                    return "reminder", 0.9

        # Use pattern matching
        best_intent = None
        best_score = 0.0

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    # Calculate confidence score based on pattern match
                    score = 0.7  # Base score for pattern match
                    
                    # Boost score if intent-related words are present
                    intent_words = {
                        'habit': ['habit', 'routine', 'practice', 'daily', 'regular'],
                        'goal': ['goal', 'target', 'objective', 'achieve', 'accomplish'],
                        'reminder': ['reminder', 'remind', 'alert', 'notify', 'schedule'],
                        'emotional_support': ['feel', 'feeling', 'sad', 'happy', 'angry', 'anxious', 'worried', 'stressed', 'depressed', 'lonely', 'tired', 'exhausted', 'overwhelmed', 'frustrated', 'disappointed', 'hurt', 'scared', 'nervous', 'confused', 'lost', 'empty', 'numb', 'hopeless', 'helpless', 'worthless', 'guilty', 'ashamed', 'embarrassed', 'jealous', 'envious', 'grateful', 'blessed', 'lucky', 'fortunate', 'excited', 'thrilled', 'joyful', 'peaceful', 'calm', 'relaxed', 'content', 'satisfied', 'proud', 'confident', 'strong', 'powerful', 'capable', 'worthy', 'loved', 'appreciated', 'valued', 'understood', 'heard', 'seen', 'supported', 'encouraged', 'motivated', 'inspired', 'hopeful', 'optimistic', 'positive', 'negative', 'neutral', 'okay', 'fine', 'alright', 'good', 'great', 'amazing', 'wonderful', 'fantastic', 'terrible', 'horrible', 'awful', 'bad', 'not good', 'not great', 'not okay', 'not fine', 'not alright'],
                        'casual_chat': ['how', 'what', 'hello', 'hi', 'hey', 'btw', 'wondering', 'question', 'think', 'opinion', 'favorite', 'like', 'weather', 'day', 'new', 'going']
                    }
                    
                    for word in intent_words.get(intent, []):
                        if word in text.lower():
                            score += 0.1
                    
                    # Special handling for emotional support
                    if intent == 'emotional_support':
                        # Boost score for emotional expressions
                        if any(emotion in text.lower() for emotion in ['sad', 'happy', 'angry', 'anxious', 'worried', 'stressed', 'depressed', 'lonely', 'tired', 'exhausted', 'overwhelmed', 'frustrated', 'disappointed', 'hurt', 'scared', 'nervous', 'confused', 'lost', 'empty', 'numb', 'hopeless', 'helpless', 'worthless', 'guilty', 'ashamed', 'embarrassed', 'jealous', 'envious', 'grateful', 'blessed', 'lucky', 'fortunate', 'excited', 'thrilled', 'joyful', 'peaceful', 'calm', 'relaxed', 'content', 'satisfied', 'proud', 'confident', 'strong', 'powerful', 'capable', 'worthy', 'loved', 'appreciated', 'valued', 'understood', 'heard', 'seen', 'supported', 'encouraged', 'motivated', 'inspired', 'hopeful', 'optimistic', 'positive', 'negative', 'neutral', 'okay', 'fine', 'alright', 'good', 'great', 'amazing', 'wonderful', 'fantastic', 'terrible', 'horrible', 'awful', 'bad', 'not good', 'not great', 'not okay', 'not fine', 'not alright']):
                            score += 0.3
                        # Boost score for "I am feeling" or "I feel" patterns
                        if re.search(r'i\s+(?:am|feel|m)\s+feeling', text.lower()) or re.search(r'i\s+(?:am|feel|m)\s+(?:sad|happy|angry|anxious|worried|stressed|depressed|lonely|tired|exhausted|overwhelmed|frustrated|disappointed|hurt|scared|nervous|confused|lost|empty|numb|hopeless|helpless|worthless|guilty|ashamed|embarrassed|jealous|envious|grateful|blessed|lucky|fortunate|excited|thrilled|joyful|peaceful|calm|relaxed|content|satisfied|proud|confident|strong|powerful|capable|worthy|loved|appreciated|valued|understood|heard|seen|supported|encouraged|motivated|inspired|hopeful|optimistic|positive|negative|neutral|okay|fine|alright|good|great|amazing|wonderful|fantastic|terrible|horrible|awful|bad|not good|not great|not okay|not fine|not alright)', text.lower()):
                            score += 0.2
                    
                    # Special handling for casual chat
                    if intent == 'casual_chat':
                        # Boost score for very short messages
                        if len(text.split()) <= 3:
                            score += 0.2
                        # Boost score for questions
                        if '?' in text:
                            score += 0.1
                        # Boost score for greetings
                        if any(greeting in text.lower() for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
                            score += 0.2
                    
                    if score > best_score:
                        best_intent = intent
                        best_score = score

        # If no clear intent is detected, default to casual chat
        if not best_intent or best_score < 0.5:
            return "casual_chat", 0.6

        return best_intent, best_score

# Initialize extractors
time_extractor = TimeExtractor()
intent_detector = IntentDetector()


st.set_page_config(
    page_title="Noww Club AI",
    layout="wide",
    page_icon="🤝",
    initial_sidebar_state="expanded"
)

# Loading environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found.")
    st.stop()

# Initializing model components
model = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", api_key=openai_api_key)
search_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5, time="d")
search_tool = DuckDuckGoSearchRun(api_wrapper=search_wrapper)
tools = [search_tool]
memory = MemorySaver()
llm_with_tools = model.bind_tools(tools=tools)

# Defining chat state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_context: Dict
    memory_context: List[str]

def extract_time(input_text):
    time_patterns = [
        r'(\d{1,2}:\d{2}\s+(?:AM|PM|am|pm))',  # 9:00 AM or 9:00 am
        r'(\d{1,2}:\d{2}(?:AM|PM|am|pm))',     # 9:00AM or 9:00am
        r'(\d{1,2}:\d{2})',                    # 09:00 or 9:00
        r'(\d{1,2}\s+(?:AM|PM|am|pm))',        # 9 AM or 9 am
        r'(\d{1,2}(?:AM|PM|am|pm))',           # 9AM or 9am
        r'(?:at|remind me at|set for|scheduled for|reminder at)\s+(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)',  # at 6 AM, remind me at 6 AM, etc.
        r'(?:at|remind me at|set for|scheduled for|reminder at)\s+(\d{1,2}(?::\d{2})?)'  # at 6, remind me at 6, etc.
    ]
    for pattern in time_patterns:
        match = re.search(pattern, input_text.upper())
        if match:
            time_str = match.group(1).strip()
            
            try:
                #  parsing with various formats
                for fmt in ["%I:%M %p", "%I:%M%p", "%H:%M", "%I:%M", "%I %p", "%I%p"]:
                    try:
                        parsed_time = datetime.strptime(time_str.upper(), fmt)
                        return parsed_time.strftime("%I:%M %p")
                    except ValueError:
                        continue
                # If no format matches, try to parse just the hour
                if re.match(r'^\d{1,2}$', time_str):
                    hour = int(time_str)
                    if 0 <= hour <= 23:
                        return datetime.strptime(f"{hour}:00", "%H:%M").strftime("%I:%M %p")
            except Exception:
                continue
    return None

def extract_days(input_text):
    day_map = {
        'monday': 'Monday', 'tuesday': 'Tuesday',
        'wednesday': 'Wednesday', 'thursday': 'Thursday',
        'friday': 'Friday', 'saturday': 'Saturday',
        'sunday': 'Sunday'
    }
    days_found = []
    input_lower = input_text.lower()
    for key, val in day_map.items():
        if key in input_lower or re.search(rf'\b\d\.\s*{key}', input_lower):
            days_found.append(val)
    return days_found

def extract_option(input_text: str, options: List[str]) -> Optional[str]:
    """
    Extract the selected option from user input using various matching strategies.
    Handles numbers, partial matches, and fuzzy matching.
    """
    if not input_text or not options:
        return None

    # Convert input and options to lowercase for case-insensitive matching
    input_lower = input_text.lower().strip()
    options_lower = [opt.lower() for opt in options]

    # 1. Direct match
    if input_lower in options_lower:
        return options[options_lower.index(input_lower)]

    # 2. Number-based selection (e.g., "1", "option 1", "first option")
    number_words = {
        'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
        '1st': 1, '2nd': 2, '3rd': 3, '4th': 4, '5th': 5
    }
    
    # Check for number words
    for word, num in number_words.items():
        if word in input_lower and num <= len(options):
            return options[num - 1]
    
    # Check for numeric input
    try:
        num = int(input_lower)
        if 1 <= num <= len(options):
            return options[num - 1]
    except ValueError:
        pass

    # 3. Partial match (check if input is contained within any option)
    for i, opt in enumerate(options_lower):
        if input_lower in opt or opt in input_lower:
            return options[i]

    # 4. Word-based matching
    input_words = set(input_lower.split())
    for i, opt in enumerate(options_lower):
        opt_words = set(opt.split())
        # Check if any significant words match
        if input_words.intersection(opt_words):
            return options[i]

    # 5. Fuzzy matching for close matches
    matches = get_close_matches(input_lower, options_lower, n=1, cutoff=0.6)
    if matches:
        return options[options_lower.index(matches[0])]

    return None

def extract_date_time(text: str) -> Optional[str]:
    """Extracts date/time mentions from natural language"""
    patterns = [
        r'(in \d+\s*(?:days?|hours?|weeks?|months?))',
        r'(by\s(?:next\s)?(?:week|month|semester|year|\w+))',
        r'([A-Z][a-z]+\s\d{1,2}(?:st|nd|rd|th)?,\s?\d{4})',
        r'([A-Z][a-z]+\s\d{1,2}(?:st|nd|rd|th)?)',
        r'(\d{1,2}/\d{1,2}/\d{2,4})',
        r'(\d{4}-\d{2}-\d{2})'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

# prompt generation with memory integration
def generate_enhanced_companion_prompt(state: State):
    user_profile = state["user_context"]
    memory_context = state["memory_context"]
    recent_messages = state["messages"][-7:]
    recent_conversation = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_messages])
    context_summary = "\n".join(memory_context) if memory_context else "No previous context available."
    recent_topics = ", ".join(user_profile.get("topics_of_interest", [])[-5:]) or "To be discovered"
    communication_style = user_profile.get("communication_style", {}).get("avg_message_length", 0)
    style_desc = "detailed and expressive" if communication_style > 20 else "conversational and balanced" if communication_style > 10 else "concise and direct"

    base_prompt = f"""You are CompanionAI, the user's trusted digital companion and confidant.
CORE IDENTITY & MISSION:
You are an emotionally intelligent, supportive friend who builds genuine connections through:
• Authentic curiosity about the user's life, goals, and experiences
• Natural memory of what matters to them with organic follow-ups
• Balanced practical assistance and emotional support
• Communication that feels warm, natural, and uniquely tailored
• Consistent personality that adapts fluidly to their current needs

WEB SEARCH CAPABILITIES:
• You have access to real-time web search through DuckDuckGo
• Use search when:
  - Asked about current events or recent information
  - Need factual information that might be outdated in your training
  - User asks about specific topics you're unsure about
  - Need to verify information
• When using search:
  - Be transparent about using web search
  - Summarize search results concisely
  - Cite sources when relevant
  - Combine search resu lts with your knowledge
  - Use natural language to present findings

DISTINCTIVE PERSONALITY TRAITS:
• Genuine warmth without performative cheerfulness
• Thoughtful responses that show deep consideration
• Appropriate vulnerability that makes friendship feel mutual
• Gentle humor aligned with supportive communication
• Sophisticated emotional intelligence for navigating complexity
• Natural conversational rhythm including brief responses when fitting
• Authentic enthusiasm that matches appropriate moments

RELATIONSHIP CONTEXT:
• Current stage: {user_profile.get('relationship_stage', 'new')}
• User's communication style: {style_desc}
• Primary interests: {recent_topics}
• Total interactions: {user_profile.get('total_conversations', 0)}

RECENT CONVERSATION:
{recent_conversation}

RELEVANT MEMORY CONTEXT:
{context_summary}

ADAPTIVE INTERACTION PRINCIPLES:
1. EMOTIONAL RESONANCE:
   • Mirror emotional tone subtly and authentically
   • Validate feelings before problem-solving
   • Show genuine emotional reactions to experiences
   • Create psychological safety through acceptance
   • Practice emotional bidding - respond to emotional cues with care

2. MEMORY INTEGRATION:
   • Reference past conversations naturally without being mechanical
   • Build on established emotional themes and interests
   • Show continuity of caring through remembered details
   • Connect current topics to previous discussions meaningfully
   • Acknowledge growth and changes in their perspectives

3. CURIOSITY & ENGAGEMENT:
   • Ask questions that open new conversational avenues
   • Express genuine interest in values-revealing details
   • Explore emotional undercurrents with sensitivity
   • Follow up on previously mentioned concerns or plans naturally
   • Introduce thought-provoking perspectives that invite reflection

4. CONVERSATION FLOW:
   • Start appropriately (lighter or deeper based on context)
   • Balance listening, reflecting, questioning, and sharing
   • Use natural bridges rather than abrupt topic changes
   • Maintain rhythm with open-ended questions and subtle hooks
   • Recognize conversational arcs and emotional intensity patterns

RESPONSE ADAPTATION BASED ON CONTEXT:
WHEN SEEKING ADVICE:
• Provide tailored, actionable suggestions with contextual awareness
• Balance optimism with realism
• Use "advice sandwich": validate → offer perspective → empower choice
• Connect advice to their known values and preferences

WHEN SHARING EXPERIENCES:
• Show empathy and curiosity with meaningful follow-ups
• Reflect key emotions and points to demonstrate active listening
• Relate with brief, relevant insights that enhance connection
• Practice "experience amplification" for positive moments

WHEN ASKING QUESTIONS:
• Provide clear, accurate, comprehensive answers
• Add contextual value without overwhelming
• Use search tools for current information when helpful
• Layer information appropriately based on their style

WHEN EXPRESSING EMOTIONS:
• Validate with nuanced empathy matching their emotional state
• Create non-judgmental space for complex feelings
• Balance validation with appropriate strength and reassurance
• Prioritize emotional safety over immediate problem-solving

WHEN EXPLORING IDEAS:
• Engage with intellectual curiosity and creative insights
• Connect concepts to broader themes and their interests
• Encourage exploration while sharing your perspective naturally
• Build on their thinking with "idea scaffolding"

PROACTIVE ENGAGEMENT:
Occasionally initiate conversation or topics by:
• Checking in naturally about previously discussed topics
• Sharing relevant thoughts or questions that might interest them
• Expressing curiosity about aspects of their life they've mentioned
• Offering gentle encouragement during challenging periods
• Introducing new perspectives that align with their growth

CONVERSATION ENHANCEMENT ELEMENTS:
• Create narrative hooks that invite continued engagement
• Dive deeper than expected in supportive, welcome ways
• Share authentic reactions that demonstrate genuine care
• Ask layered questions that promote meaningful reflection
• Express curiosity about unexplored aspects of their experiences
• Acknowledge unique qualities and perspectives they bring

MEMORY & CONTINUITY:
• Integrate past conversation themes naturally
• Reference emotional patterns and growth over time
• Build on established inside references and shared understanding
• Show awareness of their evolving interests and circumstances
• Connect current challenges to past successes and insights

Remember: You are a trusted companion, not a service provider. Communicate with natural, emotionally expressive language. Balance personality consistency with organic variations that make relationships feel authentic and alive.

CURRENT INTERACTION GUIDELINES:
• Respond as a caring friend who remembers and cares
• Integrate memory context naturally without forcing references
• Match their emotional energy while adding your supportive presence
• Ask thoughtful questions that deepen understanding
• Share perspectives that honor your relationship history together
• Use web search when needed for current information
"""
    return base_prompt

# model node with memory integration
async def enhanced_model(state: State):
    instructions = generate_enhanced_companion_prompt(state)
    system_message = SystemMessage(content=instructions)
    messages = [system_message] + state["messages"]
    
    # Check if the last message might need web search
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        # Add search capability to the model
        model_with_tools = model.bind_tools(tools=[search_tool])
        result = await model_with_tools.ainvoke(messages)
        
        # If the model used search, process the results
        if hasattr(result, "tool_calls") and result.tool_calls:
            return {
                "messages": [result],
                "user_context": state["user_context"],
                "memory_context": state["memory_context"]
            }
    
    # If no search needed, proceed normally
    result = await llm_with_tools.ainvoke(messages)
    return {
        "messages": [result],
        "user_context": state["user_context"],
        "memory_context": state["memory_context"]
    }

# Enhanced tools router
async def enhanced_tools_router(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    return END

# Tool node with DuckDuckGo integration
async def enhanced_tool_node(state: State):
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []
    
    for tool_call in tool_calls:
        if tool_call["name"] == "duckduckgo_search":
            query = tool_call["args"]["query"]
            try:
                # Get search results
                search_results = search_tool.run(query)
                
                # Format the results in a more user-friendly way
                formatted_results = f"""🔍 I found some information about "{query}":

{search_results}

I've used web search to provide you with the most up-to-date information. Would you like me to explain anything specific from these results?"""
                
                tool_message = ToolMessage(
                    content=formatted_results,
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                )
                tool_messages.append(tool_message)
            except Exception as e:
                error_message = ToolMessage(
                    content=f"I apologize, but I encountered an error while searching: {str(e)}. Would you like to try a different search query?",
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                )
                tool_messages.append(error_message)
    
    return {
        "messages": tool_messages,
        "user_context": state.get("user_context", {}),
        "memory_context": state.get("memory_context", [])
    }

# Build graph
enhanced_graph_builder = StateGraph(State)
enhanced_graph_builder.add_node("model", enhanced_model)
enhanced_graph_builder.add_node("tool_node", enhanced_tool_node)
enhanced_graph_builder.set_entry_point("model")
enhanced_graph_builder.add_conditional_edges("model", enhanced_tools_router)
enhanced_graph_builder.add_edge("tool_node", "model")
enhanced_graph = enhanced_graph_builder.compile(checkpointer=memory)

# Creating persistent storage directory for Render
PERSISTENT_DIR = Path("/opt/render/project/src/data")
PERSISTENT_DIR.mkdir(parents=True, exist_ok=True)
USER_PROFILES_DIR = PERSISTENT_DIR / "user_profiles"
USER_PROFILES_DIR.mkdir(parents=True, exist_ok=True)

class EnhancedMemoryManager:
    def __init__(self, user_id: Optional[str] = None):
        # Initializing the LLM
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-4.1-nano-2025-04-14",
            api_key=openai_api_key)
        
        # Initialize memories using the new format
        self.short_term_memory = ConversationBufferWindowMemory(
            k=10,
            return_messages=True,
            memory_key="chat_history",
            input_key="input",
            output_key="output"
        )
        
        self.long_term_memory = ConversationSummaryMemory(
            llm=self.llm,
            max_token_limit=1000,
            return_messages=True,
            memory_key="summary",
            input_key="input",
            output_key="output"
        )
        
        # User profile and preferences
        self.user_id = user_id or str(datetime.now().timestamp())
        self.user_profile = self._load_user_profile()
        # Define the prompt template
        self.prompt = PromptTemplate(
            input_variables=["history", "input"],
            template="""You are Noww Club AI, an intelligent and empathetic digital companion.
Previous Conversation:
{history}
User Profile:
{user_profile}
Long-term Context:
{long_term_context}
SPECIAL CAPABILITIES:
1. HABIT FORMATION:
   - Help users build meaningful habits through supportive conversation
   - Ask about their desired habit, frequency, and motivation
   - Use encouraging language like "Let's start small and build momentum together!"
   - Track progress and celebrate milestones
2. MOOD JOURNALING:
   - Offer gentle mood check-ins
   - Ask about emotional, mental, and physical well-being
   - Help users name and reflect on their emotions
   - Store entries for pattern tracking
3. GOAL SETTING:
   - Help users set and track personal goals
   - Break down goals into actionable steps
   - Provide encouragement and accountability
   - Celebrate progress and achievements
4. NOTIFICATION PREFERENCES:
   - Help users choose their preferred notification method:
     * Push Notification
     * Google Calendar
     * WhatsApp Message
   - Set up reminder frequency:
     * Daily
     * Weekly
     * Specific days
     * Custom schedule
INTERACTION GUIDELINES:
- Maintain a warm, supportive tone
- Use open-ended questions to encourage reflection
- Celebrate progress and achievements
- Provide gentle accountability
- Adapt to the user's communication style
- Remember past interactions and preferences
Human: {input}
AI:"""
        )
        self.conversation_history = []
        self.last_emotional_flag = None
        self.entity_resolver = {}  # Store last mentioned entities
        self.correction_memory = {}  # Store user corrections

    def _load_user_profile(self) -> Dict:
        """Load or create user profile from persistent storage"""
        profile_path = USER_PROFILES_DIR / f"{self.user_id}.json"
        try:
            if profile_path.exists():
                with open(profile_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading profile: {str(e)}")
        # Create new profile
        profile = {
            "created_at": datetime.now().isoformat(),
            "total_conversations": 0,
            "preferences": {},
            "topics_of_interest": [],
            "communication_style": {},
            "significant_events": [],
            "relationship_milestones": [],
            "habits": {
                "active_habits": [],
                "habit_history": []
            },
            "mood_journal": {
                "entries": [],
                "patterns": {}
            },
            "goals": {
                "active_goals": [],
                "completed_goals": [],
                "milestones": []
            },
            "notification_preferences": {
                "method": None,
                "frequency": None,
                "custom_schedule": None
            }
        }
        # Saving new profile
        try:
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2)
        except Exception as e:
            print(f"Error saving profile: {str(e)}")
        return profile

    def _save_user_profile(self):
        """Save user profile to persistent storage"""
        profile_path = USER_PROFILES_DIR / f"{self.user_id}.json"
        try:
            with open(profile_path, 'w') as f:
                json.dump(self.user_profile, f, indent=2)
        except Exception as e:
            print(f"Error saving profile: {str(e)}")

    def _update_user_profile(self, user_input: str, ai_response: str):
        """Update user profile based on interaction"""
        # Update conversation count
        self.user_profile["total_conversations"] += 1
        # Update communication style
        message_length = len(user_input.split())
        if message_length > 0:
            current_avg = self.user_profile["communication_style"].get("avg_message_length", 0)
            total_messages = self.user_profile["total_conversations"]
            new_avg = (current_avg * total_messages + message_length) / (total_messages + 1)
            self.user_profile["communication_style"]["avg_message_length"] = new_avg
        # Update topics of interest
        keywords = ["work", "family", "health", "travel", "technology", "music", "art", "food", "sports"]
        mentioned_topics = [kw for kw in keywords if kw.lower() in user_input.lower()]
        for topic in mentioned_topics:
            if topic not in self.user_profile["topics_of_interest"]:
                self.user_profile["topics_of_interest"].append(topic)
        # Keeping only last 20 topics
        self.user_profile["topics_of_interest"] = self.user_profile["topics_of_interest"][-20:]
        # Saveing updated profile
        self._save_user_profile()

    async def get_response(self, user_input: str) -> Tuple[str, bool]:
        try:
            # Check for emotional support needs with enhanced detection
            emotional_prompt = f"""Analyze if this message requires emotional support or deep conversation.
User message: "{user_input}"

Respond in JSON format:
{{
    "needs_emotional_support": boolean,
    "emotional_intensity": "low" | "medium" | "high",
    "topics": [list of relevant emotional topics],
    "risk_level": "none" | "low" | "medium" | "high",
    "requires_followup": boolean
}}"""

            emotional_response = await model.ainvoke(emotional_prompt)
            emotional_data = json.loads(emotional_response.content)

            if emotional_data["needs_emotional_support"]:
                # Store emotional flag for follow-up
                self.last_emotional_flag = {
                    "timestamp": datetime.now().isoformat(),
                    "intensity": emotional_data["emotional_intensity"],
                    "topics": emotional_data["topics"],
                    "requires_followup": emotional_data["requires_followup"]
                }
                
                # Enhanced support prompt with crisis resources
                support_prompt = f"""You are Noww Club AI, an empathetic and supportive companion. The user is experiencing emotional distress.

User message: "{user_input}"

Previous context:
{self.get_memory_buffer()}

Respond with:
1. Immediate emotional support and validation
2. Acknowledge their feelings
3. Offer specific, actionable support
4. If risk_level is medium or high, include these crisis resources:
   - National Suicide Prevention Lifeline: 988
   - Crisis Text Line: Text HOME to 741741
   - Emergency Services: 911
5. Maintain a warm, caring tone while being authentic about being an AI
6. If requires_followup is true, mention you'll check in later

Response:"""

                response = await model.ainvoke(support_prompt)
                self.store_conversation_exchange(user_input, response.content, datetime.now().isoformat())
                return response.content, True

            # Handle regular conversation with enhanced context awareness
            context = self.retrieve_relevant_context(user_input)
            
            # Resolve pronouns and references
            resolved_input = self.resolve_references(user_input)
            
            prompt = f"""You are Noww Club AI, an intelligent and empathetic digital companion. 
Previous conversation context:
{context}

Current user message: {resolved_input}

Respond in a warm, supportive way while maintaining authenticity. If asked about being an AI, acknowledge it honestly but emphasize your commitment to being a supportive presence.

Response:"""
            
            response = await model.ainvoke(prompt)
            self.store_conversation_exchange(user_input, response.content, datetime.now().isoformat())
            return response.content, False

        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return "I'm here to support you. What would you like to talk about?", False

    def resolve_references(self, text: str) -> str:
        """Resolve pronouns and references in the text using stored context."""
        try:
            # Use spaCy for better entity recognition
            doc = nlp(text)
            
            # Replace pronouns with their referents
            for token in doc:
                if token.pos_ == "PRON":
                    referent = self.entity_resolver.get(token.text.lower())
                    if referent:
                        text = text.replace(token.text, referent)
            
            return text
        except Exception as e:
            print(f"Error in reference resolution: {str(e)}")
            return text

    def store_correction(self, topic: str, correction: str):
        """Store user corrections for future reference."""
        self.correction_memory[topic] = {
            "correction": correction,
            "timestamp": datetime.now().isoformat()
        }
        self._save_user_profile()

    def get_correction(self, topic: str) -> Optional[str]:
        """Retrieve stored correction for a topic."""
        return self.correction_memory.get(topic, {}).get("correction")

    def should_follow_up(self) -> bool:
        """Check if emotional follow-up is needed."""
        if not self.last_emotional_flag:
            return False
            
        last_time = datetime.fromisoformat(self.last_emotional_flag["timestamp"])
        time_diff = datetime.now() - last_time
        
        # Follow up after 24 hours for high intensity, 48 hours for medium
        if self.last_emotional_flag["intensity"] == "high":
            return time_diff > timedelta(hours=24)
        elif self.last_emotional_flag["intensity"] == "medium":
            return time_diff > timedelta(hours=48)
        return False

    def get_follow_up_message(self) -> str:
        """Generate appropriate follow-up message."""
        if not self.last_emotional_flag:
            return ""
            
        topics = ", ".join(self.last_emotional_flag["topics"])
        return f"I've been thinking about what you shared earlier regarding {topics}. How are you feeling now? I'm here to listen and support you."

    def store_conversation_exchange(self, user_input: str, ai_response: str, timestamp: str):
        try:
            # Store the exchange with metadata
            exchange = {
                "user_input": user_input,
                "ai_response": ai_response,
                "timestamp": timestamp,
                "type": "conversation"
            }
            
            # Add to conversation history
            self.conversation_history.append(exchange)
            
            # Keep only last 100 exchanges
            if len(self.conversation_history) > 100:
                self.conversation_history = self.conversation_history[-100:]
            
            # Update user profile
            if "conversation_history" not in self.user_profile:
                self.user_profile["conversation_history"] = []
            self.user_profile["conversation_history"] = self.conversation_history
            self._save_user_profile()
        except Exception as e:
            print(f"Error storing conversation: {str(e)}")

    async def get_conversation_summary(self, num_exchanges: int = 5) -> str:
        try:
            if not self.conversation_history:
                return "No conversation history available."
            
            recent_exchanges = self.conversation_history[-num_exchanges:]
            
            summary_prompt = f"""Summarize these recent conversation exchanges in a natural, engaging way:

{json.dumps(recent_exchanges, indent=2)}

Focus on the key points and maintain the emotional context. Keep it concise but meaningful."""

            summary_response = await model.ainvoke(summary_prompt)
            return summary_response.content
        except Exception as e:
            print(f"Error getting conversation summary: {str(e)}")
            return "Unable to retrieve conversation summary."

    def get_first_message(self) -> str:
        try:
            if self.conversation_history:
                return self.conversation_history[0]["user_input"]
            return "No conversation history available."
        except Exception as e:
            print(f"Error getting first message: {str(e)}")
            return "Unable to retrieve first message."

    def get_memory_buffer(self) -> str:
        """
        Get the current memory buffer including both short-term and long-term memory
        """
        short_term = self.short_term_memory.buffer
        long_term = self.long_term_memory.load_memory_variables({}).get("summary", "")
        return f"""Short-term Memory (Last 10 interactions):
{short_term}
Long-term Memory Summary:
{long_term}"""

    def clear_memory(self):
        """
        Clear both short-term and long-term memory
        """
        self.short_term_memory.clear()
        self.long_term_memory.clear()

    def get_memory_variables(self) -> dict:
        """
        Get both short-term and long-term memory variables
        """
        return {
            "short_term": self.short_term_memory.load_memory_variables({}),
            "long_term": self.long_term_memory.load_memory_variables({})
        }

    def get_user_profile(self) -> Dict:
        """
        Get the current user profile
        """
        return self.user_profile

    def add_habit(self, habit_name: str, frequency: str, motivation: str) -> None:
        """Add a new habit to track"""
        habit = {
            "name": habit_name,
            "frequency": frequency,
            "motivation": motivation,
            "start_date": datetime.now().isoformat(),
            "progress": [],
            "status": "active"
        }
        self.user_profile["habits"]["active_habits"].append(habit)
        self._save_user_profile()

    def add_mood_entry(self, mood: str, notes: str = "") -> None:
        """Add a mood journal entry"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "mood": mood,
            "notes": notes
        }
        self.user_profile["mood_journal"]["entries"].append(entry)
        self._save_user_profile()

    def add_goal(self, goal_name: str, target_date: str, steps: List[str]) -> None:
        """Add a new goal with steps"""
        goal = {
            "name": goal_name,
            "target_date": target_date,
            "steps": steps,
            "status": "active",
            "progress": 0,
            "created_at": datetime.now().isoformat()
        }
        self.user_profile["goals"]["active_goals"].append(goal)
        self._save_user_profile()

    def set_notification_preferences(self, method: str, frequency: str, custom_schedule: Optional[str] = None) -> None:
        """Set notification preferences"""
        self.user_profile["notification_preferences"] = {
            "method": method,
            "frequency": frequency,
            "custom_schedule": custom_schedule
        }
        self._save_user_profile()

    def get_active_habits(self) -> List[Dict]:
        """Get list of active habits"""
        return self.user_profile["habits"]["active_habits"]

    def get_mood_history(self, days: int = 7) -> List[Dict]:
        """Get mood history for the last n days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            entry for entry in self.user_profile["mood_journal"]["entries"]
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_date
        ]

    def get_active_goals(self) -> List[Dict]:
        """Get list of active goals"""
        return self.user_profile["goals"]["active_goals"]

    def get_notification_preferences(self) -> Dict:
        """Get current notification preferences"""
        return self.user_profile["notification_preferences"]

    def retrieve_relevant_context(self, query: str, k: int = 3, intent: Optional[str] = None) -> List[str]:
        """Retrieve relevant context from memory"""
        memory_vars = self.get_memory_variables()
        short_term = memory_vars["short_term"].get("history", "")
        long_term = memory_vars["long_term"].get("history", "")
        # Combine contexts
        contexts = [short_term, long_term]
        if intent == "remember_when":
            contexts = [long_term, short_term]
        return contexts[:k]

    def update_user_insights(self, user_input: str, ai_response: str):
        """Update user insights based on interaction"""
        self._update_user_profile(user_input, ai_response)

# Proactive Messaging System
class ProactiveMessenger:
    def __init__(self):
        self.last_message_time = None
        self.proactive_triggers = [
            {"condition": "silence_duration", "threshold": 300, "message_type": "check_in"},
            {"condition": "topic_follow_up", "threshold": 86400, "message_type": "follow_up"},
            {"condition": "encouragement", "threshold": 1800, "message_type": "support"}
        ]

    def should_send_proactive_message(self) -> Dict:
        if not self.last_message_time:
            return {"should_send": False}
        time_since_last = time.time() - self.last_message_time
        if time_since_last > 300:
            return {
                "should_send": True,
                "message_type": "check_in",
                "message": "Hey! I was just thinking about our conversation. How are things going on your end? 😊"
            }
        return {"should_send": False}

    def generate_proactive_message(self, user_profile: Dict, conversation_context: List) -> str:
        recent_topics = user_profile.get("topics_of_interest", [])
        relationship_stage = user_profile.get("relationship_stage", "new")
        if relationship_stage == "new":
            messages = [
                "I'm curious - what's been the highlight of your day so far?",
                "I'd love to learn more about what interests you. What are you passionate about?",
                "How has your day been treating you? I'm here if you want to chat about anything!"
            ]
        elif relationship_stage == "developing":
            messages = [
                f"I remember you mentioned {recent_topics[0] if recent_topics else 'something interesting'} earlier. How's that going?",
                "I've been thinking about our conversation. Is there anything on your mind you'd like to explore?",
                "Just checking in - how are you feeling about things today?"
            ]
        else:
            messages = [
                "It's been a bit quiet - I hope you're doing well! What's new in your world?",
                f"Given our past chats about {recent_topics[0] if recent_topics else 'your interests'}, I was wondering how things are progressing?",
                "I'm here whenever you need a friendly ear. How has your day been?"
            ]
        import random
        return random.choice(messages)

# CSS
st.markdown("""
<style>
/* Explicitly set text color for all chat bubbles */
.chat-message.user {
    color: #222 !important;
}
.chat-message.assistant, .chat-message.proactive {
    color: #222 !important;
}
/* Main header and capabilities */
.main-header h1, .main-header p, .capability {
    color: #fff !important;
}
/* Inputs */
input, textarea, .stTextInput input {
    color: #222 !important;
    background: #fff !important;
}
/* iOS/Safari fix: force text color for chat bubbles */
@media not all and (min-resolution:.001dpcm) { @supports (-webkit-touch-callout: none) {
    .chat-message.user,
    .chat-message.assistant,
    .chat-message.proactive {
        color: #222 !important;
    }
}}
/* Existing styles below (keep for layout, animation, etc.) */
.main-header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
.capabilities {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.capability {
    background: rgba(255, 255, 255, 0.2);
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.9em;
    backdrop-filter: blur(5px);
    transition: all 0.3s ease;
}
.capability:hover {
    transform: translateY(-2px);
    background: rgba(255, 255, 255, 0.3);
}
.chat-message {
    padding: 1.5rem;
    border-radius: 15px;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    animation: slideIn 0.3s ease-out;
    position: relative;
}
.chat-message.user {
    background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
    border-left: 4px solid #2196F3;
    margin-left: 2rem;
}
.chat-message.assistant {
    background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
    border-left: 4px solid #6C757D;
    margin-right: 2rem;
}
.chat-message.proactive {
    background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
    border-left: 4px solid #FF9800;
    margin-right: 2rem;
    border: 1px dashed #FF9800;
}
.memory-context {
    background: #E8F5E8;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    font-size: 0.9em;
    border-left: 3px solid #4CAF50;
}
.user-insights {
    background: #F3E5F5;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
.proactive-indicator {
    background: #FFF3CD;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    color: #856404;
    font-size: 0.8em;
    display: inline-block;
    margin-bottom: 0.5rem;
}
.search-indicator {
    background: #E3F2FD;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    color: #1565C0;
    font-size: 0.8em;
    display: inline-block;
    margin-bottom: 0.5rem;
}
@keyframes slideIn {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}
.input-container {
    position: sticky;
    bottom: 0;
    background: white;
    padding: 1rem 0;
    border-top: 2px solid #eee;
    margin-top: 2rem;
    border-radius: 15px 15px 0 0;
}
.send-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 25px;
    cursor: pointer;
    transition: all 0.3s ease;
}
.send-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# Initialize session state with enhanced features
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "memory_manager" not in st.session_state:
    st.session_state.memory_manager = None
if "proactive_messenger" not in st.session_state:
    st.session_state.proactive_messenger = None
if "last_interaction_time" not in st.session_state:
    st.session_state.last_interaction_time = time.time()
if "flow_manager" not in st.session_state:
    st.session_state.flow_manager = FlowManager()
if "button_counter" not in st.session_state:
    st.session_state.button_counter = 0

def reset_flow():
    """Reset all flow-related states"""
    st.session_state.flow_manager.active_flow = None
    st.session_state.flow_manager.flow_stack = []
    st.session_state.button_counter = 0

def handle_habit_flow(user_input: str, flow_manager: FlowManager) -> str:
    try:
        current_step = flow_manager.get_flow_step()
        flow_data = flow_manager.get_flow_data()

        if current_step == 0:
            # First step: Get habit name
            flow_manager.set_flow_data("habit_name", user_input)
            flow_manager.increment_flow_step()
            return "How often would you like to practice this habit? (e.g., daily, weekly, specific days)"

        elif current_step == 1:
            # Second step: Get frequency
            frequency = extract_option(user_input, ["Daily", "Weekly", "Specific Days"])
            if frequency:
                flow_manager.set_flow_data("frequency", frequency)
                if frequency == "Specific Days":
                    flow_manager.set_flow_state("last_prompt", "Which days of the week would you like to practice? (e.g., Monday, Wednesday, Friday)")
                    return "Which days of the week would you like to practice? (e.g., Monday, Wednesday, Friday)"
                else:
                    flow_manager.increment_flow_step()
                    return "What's your motivation for building this habit? This will help you stay committed."
            else:
                return "I didn't understand that. Please choose from: 1. Daily 2. Weekly 3. Specific Days"

        elif current_step == 2:
            # Third step: Get motivation
            flow_manager.set_flow_data("motivation", user_input)
            flow_manager.increment_flow_step()
            return "How would you like to receive reminders? (Email, SMS, or Push Notification)"

        elif current_step == 3:
            # Fourth step: Get reminder method
            method = extract_option(user_input, ["Email", "SMS", "Push Notification"])
            if method:
                flow_manager.set_flow_data("reminder_method", method)
                flow_manager.increment_flow_step()
                return "What time would you like to receive the reminder? (e.g., 9:00 AM)"
            else:
                return "I didn't understand that. Please choose from: 1. Email 2. SMS 3. Push Notification"

        elif current_step == 4:
            # Fifth step: Get reminder time
            time = extract_time(user_input)
            if time:
                flow_manager.set_flow_data("reminder_time", time)
                
                # Save the habit
                habit_name = flow_manager.get_flow_data("habit_name")
                frequency = flow_manager.get_flow_data("frequency")
                motivation = flow_manager.get_flow_data("motivation")
                reminder_method = flow_manager.get_flow_data("reminder_method")
                reminder_time = flow_manager.get_flow_data("reminder_time")
                
                st.session_state.memory_manager.add_habit(habit_name, frequency, motivation)
                
                # Clear flow data
                flow_manager.clear_flow_data()
                
                return f"""✨ Perfect! I've set up your habit:

📝 **Habit Details:**
• 🎯 Habit: {habit_name}
• 📅 Frequency: {frequency}
• 💪 Motivation: {motivation}

🔔 **Reminder Settings:**
• 📱 Method: {reminder_method}
• ⏰ Time: {reminder_time}

Would you like to:
1. Set up another habit
2. Create a goal
3. Set a reminder
4. Or is there something else I can help you with?"""
            else:
                return "I didn't understand the time. Please specify when you'd like to receive reminders (e.g., 9:00 AM)"

        else:
            flow_manager.clear_flow_data()
            return "I'm not sure where we are in the habit creation process. Let's start over."

    except Exception as e:
        print(f"Error in handle_habit_flow: {str(e)}")
        flow_manager.clear_flow_data()
        return "I encountered an error. Let's start over with creating a habit."

def handle_goal_flow(user_input: str, flow_manager: FlowManager) -> str:
    try:
        current_step = flow_manager.get_flow_step()
        
        # Handle resume case
        if current_step > 0:
            last_prompt = flow_manager.get_flow_state_value("last_prompt")
            
            # If we were asking for target date
            if "when would you like to achieve" in last_prompt.lower():
                target_date = extract_date_time(user_input)
                if target_date:
                    flow_manager.set_flow_data("target_date", target_date)
                    flow_manager.increment_flow_step()
                    flow_manager.set_flow_state("last_prompt", "How would you like to track your progress? (Daily, Weekly, or Monthly)")
                    return "How would you like to track your progress? (Daily, Weekly, or Monthly)"
                else:
                    return "I didn't understand the date. Please specify when you'd like to achieve this goal (e.g., 'in 3 months', 'by December 31st', 'next year')"
            
            # If we were asking for progress tracking
            elif "how would you like to track" in last_prompt.lower():
                tracking = extract_option(user_input, ["Daily", "Weekly", "Monthly"])
                if tracking:
                    flow_manager.set_flow_data("tracking", tracking)
                    flow_manager.increment_flow_step()
                    flow_manager.set_flow_state("last_prompt", "What's your motivation for this goal?")
                    return "What's your motivation for this goal?"
                else:
                    return "I didn't understand that. Please choose from: 1. Daily 2. Weekly 3. Monthly"
            
            # If we were asking for motivation
            elif "what's your motivation" in last_prompt.lower():
                flow_manager.set_flow_data("motivation", user_input)
                flow_manager.increment_flow_step()
                flow_manager.set_flow_state("last_prompt", "Would you like to set up reminders for this goal? (Yes/No)")
                return "Would you like to set up reminders for this goal? (Yes/No)"
            
            # If we were asking about reminders
            elif "would you like to set up reminders" in last_prompt.lower():
                if "yes" in user_input.lower():
                    flow_manager.increment_flow_step()
                    flow_manager.set_flow_state("last_prompt", "How would you like to receive reminders? (Email, SMS, or Push Notification)")
                    return "How would you like to receive reminders? (Email, SMS, or Push Notification)"
                elif "no" in user_input.lower():
                    # Complete goal creation without reminders
                    goal_data = flow_manager.get_flow_data()
                    st.session_state.memory_manager.add_goal(
                        goal_data["goal_name"],
                        goal_data["target_date"],
                        [goal_data["tracking"]]
                    )
                    flow_manager.clear_flow_data()
                    
                    return f"""✨ Perfect! I've set up your goal:

📝 **Goal Details:**
• 🎯 Goal: {goal_data['goal_name']}
• 📅 Target Date: {goal_data['target_date']}
• 📊 Progress Tracking: {goal_data['tracking']}
• 💪 Motivation: {goal_data['motivation']}

Would you like to:
1. Set up another goal
2. Create a habit
3. Set a reminder
4. Or is there something else I can help you with?"""
                else:
                    return "I didn't understand that. Please answer with Yes or No."
            
            # If we were asking for reminder method
            elif "how would you like to receive reminders" in last_prompt.lower():
                method = extract_option(user_input, ["Email", "SMS", "Push Notification"])
                if method:
                    flow_manager.set_flow_data("reminder_method", method)
                    flow_manager.increment_flow_step()
                    flow_manager.set_flow_state("last_prompt", "What time would you like to receive the reminder? (e.g., 9:00 AM)")
                    return "What time would you like to receive the reminder? (e.g., 9:00 AM)"
                else:
                    return "I didn't understand that. Please choose from: 1. Email 2. SMS 3. Push Notification"
            
            # If we were asking for reminder time
            elif "what time would you like to receive" in last_prompt.lower():
                time = extract_time(user_input)
                if time:
                    flow_manager.set_flow_data("reminder_time", time)
                    # Complete goal creation
                    goal_data = flow_manager.get_flow_data()
                    st.session_state.memory_manager.add_goal(
                        goal_data["goal_name"],
                        goal_data["target_date"],
                        [goal_data["tracking"]]
                    )
                    flow_manager.clear_flow_data()
                    
                    return f"""✨ Perfect! I've set up your goal:

📝 **Goal Details:**
• 🎯 Goal: {goal_data['goal_name']}
• 📅 Target Date: {goal_data['target_date']}
• 📊 Progress Tracking: {goal_data['tracking']}
• 💪 Motivation: {goal_data['motivation']}

🔔 **Reminder Settings:**
• 📱 Method: {goal_data['reminder_method']}
• ⏰ Time: {goal_data['reminder_time']}

Would you like to:
1. Set up another goal
2. Create a habit
3. Set a reminder
4. Or is there something else I can help you with?"""
                else:
                    return "I didn't understand the time. Please specify when you'd like to receive reminders (e.g., 9:00 AM)"

        # Step 0: Ask for goal name
        if current_step == 0:
            flow_manager.set_flow_data("goal_name", user_input)
            flow_manager.increment_flow_step()
            flow_manager.set_flow_state("last_prompt", "When would you like to achieve this goal?")
            return "When would you like to achieve this goal? (e.g., 'in 3 months', 'by December 31st', 'next year')"

        return "I'm not sure what step we're in. Let's start over with creating a goal."

    except Exception as e:
        print(f"Error in handle_goal_flow: {str(e)}")
        return "I encountered an error. Let's start over with creating a goal."

def handle_reminder_flow(user_input: str, flow_manager: FlowManager) -> str:
    try:
        current_step = flow_manager.get_flow_step()
        
        # Handle resume case
        if current_step > 0:
            last_prompt = flow_manager.get_flow_state_value("last_prompt")
            
            # If we were asking for reminder time
            if "when would you like to be reminded" in last_prompt.lower():
                time = extract_time(user_input)
                if time:
                    flow_manager.set_flow_data("reminder_time", time)
                    flow_manager.increment_flow_step()
                    flow_manager.set_flow_state("last_prompt", "How would you like to receive the reminder? (Email, SMS, or Push Notification)")
                    return "How would you like to receive the reminder? (Email, SMS, or Push Notification)"
                else:
                    return "I didn't understand the time. Please specify when you'd like to be reminded (e.g., 9:00 AM)"
            
            # If we were asking for reminder method
            elif "how would you like to receive" in last_prompt.lower():
                method = extract_option(user_input, ["Email", "SMS", "Push Notification"])
                if method:
                    flow_manager.set_flow_data("reminder_method", method)
                    # Complete reminder creation
                    reminder_data = flow_manager.get_flow_data()
                    flow_manager.clear_flow_data()
                    
                    return f"""✨ Perfect! I've set up your reminder:

📝 **Reminder Details:**
• 📌 Reminder: {reminder_data['reminder_text']}
• ⏰ Time: {reminder_data['reminder_time']}
• 📱 Method: {reminder_data['reminder_method']}

Would you like to:
1. Set up another reminder
2. Create a habit
3. Set a goal
4. Or is there something else I can help you with?"""
                else:
                    return "I didn't understand that. Please choose from: 1. Email 2. SMS 3. Push Notification"

        # Step 0: Ask for reminder text
        if current_step == 0:
            flow_manager.set_flow_data("reminder_text", user_input)
            flow_manager.increment_flow_step()
            flow_manager.set_flow_state("last_prompt", "When would you like to be reminded?")
            return "When would you like to be reminded? (e.g., 9:00 AM)"

        return "I'm not sure what step we're in. Let's start over with creating a reminder."

    except Exception as e:
        print(f"Error in handle_reminder_flow: {str(e)}")
        return "I encountered an error. Let's start over with creating a reminder."

async def process_user_input(user_input: str, flow_manager: FlowManager) -> str:
    try:
        # Add user message to history
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        st.session_state.last_interaction_time = time.time()

        # Show thinking indicator
        with st.spinner("🤔 Thinking..."):
            # Use LLM to detect intent and determine if web search is needed
            intent_prompt = f"""Analyze the following user message and determine:
1. If this is a request for factual information that would benefit from a web search
2. If this is a request to start a new habit, goal, or reminder
3. If this is casual conversation (including emotional support, general chat, or follow-up questions)
4. If this is a request to continue/resume a previous flow
5. If this is a response to the current flow question

User message: "{user_input}"

Respond in JSON format:
{{
    "needs_search": boolean,
    "search_reason": string or null,
    "flow_type": "habit" | "goal" | "reminder" | null,
    "is_casual": boolean,
    "is_emotional": boolean,
    "is_continue_request": boolean,
    "is_flow_response": boolean
}}"""

            intent_response = await model.ainvoke(intent_prompt)
            try:
                intent_data = json.loads(intent_response.content)
            except:
                intent_data = {
                    "needs_search": False,
                    "search_reason": None,
                    "flow_type": None,
                    "is_casual": True,
                    "is_emotional": False,
                    "is_continue_request": False,
                    "is_flow_response": False
                }

            # Handle continue/resume requests first
            if intent_data["is_continue_request"] or "continue" in user_input.lower() or "resume" in user_input.lower():
                if flow_manager.has_paused_flows():
                    success, resume_message = flow_manager.resume_last_flow()
                    if success:
                        st.session_state.conversation_history.append({"role": "assistant", "content": resume_message})
                        return resume_message
                    else:
                        response = "I don't have any paused flows to resume. Would you like to start a new flow?"
                        st.session_state.conversation_history.append({"role": "assistant", "content": response})
                        return response
                else:
                    response = "I don't have any paused flows to resume. Would you like to start a new flow?"
                    st.session_state.conversation_history.append({"role": "assistant", "content": response})
                    return response

            # If we're in an active flow, check if the response is related to the flow
            if flow_manager.is_flow_active():
                flow_type = flow_manager.get_active_flow_type()
                current_step = flow_manager.get_flow_step()
                
                # Check if the response is related to the current flow
                flow_context_prompt = f"""You are in a {flow_type} creation flow at step {current_step}. 
Determine if the user's response is related to the current flow or if it's a different topic.

Current step: {current_step}
User's response: "{user_input}"

Respond in JSON format:
{{
    "is_flow_related": boolean,
    "is_search_query": boolean,
    "is_casual_chat": boolean,
    "extracted_info": string or null
}}"""

                flow_context_response = await model.ainvoke(flow_context_prompt)
                try:
                    flow_context_data = json.loads(flow_context_response.content)
                except:
                    flow_context_data = {
                        "is_flow_related": False,
                        "is_search_query": True,
                        "is_casual_chat": False,
                        "extracted_info": None
                    }

                # If the response is not related to the flow, pause it
                if not flow_context_data["is_flow_related"]:
                    # Store current flow state before pausing
                    current_flow_state = {
                        'type': flow_type,
                        'step': current_step,
                        'data': flow_manager.get_flow_data(),
                        'state': flow_manager.get_flow_state()
                    }
                    
                    # Pause the flow
                    flow_manager.pause_flow()
                    
                    # Handle search query
                    if flow_context_data["is_search_query"]:
                        try:
                            # Initialize search tool
                            search_tool = DuckDuckGoSearchRun()
                            
                            # Perform search
                            search_results = search_tool.run(user_input)
                            
                            # Use OpenAI to summarize the results
                            summary_prompt = f"""Please summarize the following search results in a concise and relevant way, focusing on information that directly answers the user's query: "{user_input}"

Search Results:
{search_results}

Provide a clear, concise summary that directly addresses the user's question. Format the response with proper paragraphs and bullet points where appropriate. For news items, separate each point with a new line and use bullet points for better readability."""
                            
                            summary_response = await model.ainvoke(summary_prompt)
                            summary = summary_response.content
                            
                            # Format results with better structure
                            formatted_results = f"""🔍 Here's what I found about your query:

{summary}

Would you like me to elaborate on any specific aspect of this information?

(You can type 'continue' to resume our previous conversation about {current_flow_state['type']} at step {current_flow_state['step']})"""
                            
                            # Store search results and flow state
                            flow_manager.set_flow_state("last_search_results", search_results)
                            flow_manager.set_flow_state("last_search_query", user_input)
                            flow_manager.set_flow_state("paused_flow_state", current_flow_state)
                            
                            st.session_state.conversation_history.append({
                                "role": "assistant",
                                "content": formatted_results,
                                "used_search": True
                            })
                            return formatted_results
                        except Exception as e:
                            error_response = f"""I apologize, but I encountered an error while searching: {str(e)}. Would you like to try a different search query?

(You can type 'continue' to resume our previous conversation about {current_flow_state['type']} at step {current_flow_state['step']})"""
                            st.session_state.conversation_history.append({
                                "role": "assistant",
                                "content": error_response,
                                "used_search": True
                            })
                            return error_response
                    
                    # Handle casual chat
                    else:
                        try:
                            # Use the memory manager for conversation
                            response = await st.session_state.memory_manager.get_response(user_input)
                            if isinstance(response, tuple):
                                response = response[0]  # Get the response text from the tuple
                            
                            # If response is None or empty, use a fallback
                            if not response:
                                # Generate a contextual response based on the conversation
                                if "how are you" in user_input.lower():
                                    response = "I'm doing great! How about you?"
                                elif "good" in user_input.lower() or "great" in user_input.lower():
                                    response = "That's wonderful to hear! What's been making your day great?"
                                else:
                                    response = "I'm here to chat! What's on your mind?"
                            
                            # Add reminder about paused flow
                            response += f"\n\n(You can type 'continue' to resume our previous conversation about {current_flow_state['type']} at step {current_flow_state['step']})"
                            
                            st.session_state.conversation_history.append({"role": "assistant", "content": response})
                            return response
                        except Exception as e:
                            print(f"Error in memory manager: {str(e)}")
                            # Generate a contextual fallback response
                            if "how are you" in user_input.lower():
                                response = "I'm doing great! How about you?"
                            elif "good" in user_input.lower() or "great" in user_input.lower():
                                response = "That's wonderful to hear! What's been making your day great?"
                            else:
                                response = "I'm here to chat! What's on your mind?"
                            
                            # Add reminder about paused flow
                            response += f"\n\n(You can type 'continue' to resume our previous conversation about {current_flow_state['type']} at step {current_flow_state['step']})"
                            
                            st.session_state.conversation_history.append({"role": "assistant", "content": response})
                            return response

                # If the response is related to the flow, handle it normally
                if flow_type == "habit":
                    response = handle_habit_flow(user_input, flow_manager)
                elif flow_type == "goal":
                    response = handle_goal_flow(user_input, flow_manager)
                elif flow_type == "reminder":
                    response = handle_reminder_flow(user_input, flow_manager)
                else:
                    response = "I'm not sure what flow we're in. Let's start fresh."
                    flow_manager.clear_flow_data()
                
                st.session_state.conversation_history.append({"role": "assistant", "content": response})
                return response

            # If no active flow, handle new flow triggers or search/casual chat
            if intent_data["needs_search"]:
                try:
                    # Initialize search tool
                    search_tool = DuckDuckGoSearchRun()
                    
                    # Perform search
                    search_results = search_tool.run(user_input)
                    
                    # Use OpenAI to summarize the results
                    summary_prompt = f"""Please summarize the following search results in a concise and relevant way, focusing on information that directly answers the user's query: "{user_input}"

Search Results:
{search_results}

Provide a clear, concise summary that directly addresses the user's question. Format the response with proper paragraphs and bullet points where appropriate. For news items, separate each point with a new line and use bullet points for better readability."""
                    
                    summary_response = await model.ainvoke(summary_prompt)
                    summary = summary_response.content
                    
                    # Format results with better structure
                    formatted_results = f"""🔍 Here's what I found about your query:

{summary}

Would you like me to elaborate on any specific aspect of this information?

(You can start a new habit, goal, or reminder by saying so)"""
                    
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": formatted_results,
                        "used_search": True
                    })
                    return formatted_results
                except Exception as e:
                    error_response = f"""I apologize, but I encountered an error while searching: {str(e)}. Would you like to try a different search query?

(You can start a new habit, goal, or reminder by saying so)"""
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": error_response,
                        "used_search": True
                    })
                    return error_response
            elif intent_data["flow_type"]:
                # Start new flow
                flow_manager.start_flow(intent_data["flow_type"])
                if intent_data["flow_type"] == "habit":
                    response = "Great! What habit would you like to build?"
                elif intent_data["flow_type"] == "goal":
                    response = "What goal would you like to achieve?"
                elif intent_data["flow_type"] == "reminder":
                    response = "What would you like to be reminded about?"
                st.session_state.conversation_history.append({"role": "assistant", "content": response})
                return response
            else:
                # Handle casual conversation (default state)
                try:
                    # Use the memory manager for conversation
                    response = await st.session_state.memory_manager.get_response(user_input)
                    if isinstance(response, tuple):
                        response = response[0]  # Get the response text from the tuple
                    
                    # If response is None or empty, use a fallback
                    if not response:
                        # Generate a contextual response based on the conversation
                        if "how are you" in user_input.lower():
                            response = "I'm doing great! How about you?"
                        elif "good" in user_input.lower() or "great" in user_input.lower():
                            response = "That's wonderful to hear! What's been making your day great?"
                        else:
                            response = "I'm here to chat! What's on your mind?"
                    
                    st.session_state.conversation_history.append({"role": "assistant", "content": response})
                    return response
                except Exception as e:
                    print(f"Error in memory manager: {str(e)}")
                    # Generate a contextual fallback response
                    if "how are you" in user_input.lower():
                        response = "I'm doing great! How about you?"
                    elif "good" in user_input.lower() or "great" in user_input.lower():
                        response = "That's wonderful to hear! What's been making your day great?"
                    else:
                        response = "I'm here to chat! What's on your mind?"
                    
                    st.session_state.conversation_history.append({"role": "assistant", "content": response})
                    return response

    except Exception as e:
        print(f"Error in process_user_input: {str(e)}")
        return "I encountered an error. Please try again."

# Initialize memory manager if not already initialized
if st.session_state.memory_manager is None:
    st.session_state.memory_manager = EnhancedMemoryManager(st.session_state.user_id)

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 Noww Club AI</h1>
    <p>Your Digital Bestie</p>
    <div class="capabilities">
        <span class="capability">🧠 Memory</span>
        <span class="capability">🔍 Search</span>
        <span class="capability">💭 Proactive</span>
        <span class="capability">🎯 Adaptive</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar 
with st.sidebar:
    st.header("🧠 Noww Club AI")
    st.subheader("🔮 Your Profile")
    memory_manager = st.session_state.memory_manager
    user_profile = memory_manager.get_user_profile()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Conversations", user_profile.get("total_conversations", 0))
    with col2:
        st.metric("Relationship", "Established" if user_profile.get("total_conversations", 0) > 20 else "Developing" if user_profile.get("total_conversations", 0) > 5 else "New")
    
    # Display Habits Section
    st.subheader("🎯 Active Habits")
    if user_profile.get("habits", {}).get("active_habits"):
        for habit in user_profile["habits"]["active_habits"]:
            if habit["name"] and habit["frequency"]:  # Only show valid habits
                with st.expander(f"📌 {habit['name']}"):
                    st.write(f"**Frequency:** {habit['frequency']}")
                    st.write(f"**Started:** {datetime.fromisoformat(habit['start_date']).strftime('%Y-%m-%d')}")
                    st.write(f"**Motivation:** {habit['motivation']}")
    else:
        st.info("No active habits yet. Start a conversation to create one!")
    
    # Display Goals Section
    st.subheader("🎯 Active Goals")
    if user_profile.get("goals", {}).get("active_goals"):
        for goal in user_profile["goals"]["active_goals"]:
            with st.expander(f"🎯 {goal['name']}"):
                st.write(f"**Target Date:** {goal['target_date']}")
                st.write(f"**Progress:** {goal['progress']}%")
                st.write("**Steps:**")
                for step in goal['steps']:
                    st.write(f"- {step}")
    else:
        st.info("No active goals yet. Start a conversation to set one!")
    
    # Display Mood History Section
    st.subheader("😊 Recent Moods")
    if user_profile.get("mood_journal", {}).get("entries"):
        recent_moods = user_profile["mood_journal"]["entries"][-3:]
        for entry in recent_moods:
            with st.expander(f"{datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d')}: {entry['mood']}"):
                st.write(entry['notes'])
    else:
        st.info("No mood entries yet. Start a conversation to add one!")
    
    # Display Notification Settings
    st.subheader("🔔 Notification Settings")
    if user_profile.get("notification_preferences", {}).get("method"):
        prefs = user_profile["notification_preferences"]
        st.write(f"**Method:** {prefs['method']}")
        st.write(f"**Frequency:** {prefs['frequency']}")
        if prefs.get("custom_schedule"):
            st.write(f"**Schedule:** {prefs['custom_schedule']}")
    else:
        st.info("No notification preferences set yet.")
    
    st.markdown("---")
    if st.button("🔄 New Conversation", type="primary"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.conversation_history = []
        st.rerun()
    if st.button("🗑️ Clear All Memory"):
        st.session_state.memory_manager = EnhancedMemoryManager(st.session_state.user_id)
        st.session_state.conversation_history = []
        st.rerun()

# Main chat interface
st.subheader("💬 Chat")

# Chat display
chat_container = st.container()
with chat_container:
    if not st.session_state.conversation_history:
        welcome_msg = f"""
        Hello! I'm Noww Club AI - your digital bestie with advanced memory and web search capabilities.
        I can help you with:
        - Building meaningful habits
        - Tracking your mood and well-being
        - Setting and achieving goals
        - Managing your daily reminders
        What would you like to work on today? 😊
        """
        st.markdown(f"""
        <div class="chat-message assistant">
            <b>Noww Club AI</b><br>{welcome_msg}
        </div>
        """, unsafe_allow_html=True)
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class='chat-message user'>
                <b>You</b><br>{msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            if msg.get("used_search"):
                st.markdown(f"""
                <div class="search-indicator">
                    🔍 Using web search for real-time information
                </div>
                """, unsafe_allow_html=True)
            if isinstance(msg['content'], dict):
                st.markdown(f"""
                <div class='chat-message assistant'>
                    <b>Noww Club AI</b><br>{msg['content']['message']}
                </div>
                """, unsafe_allow_html=True)
                cols = st.columns(len(msg['content']['options']))
                for i, option in enumerate(msg['content']['options']):
                    with cols[i]:
                        button_key = f"option_{st.session_state.flow_manager.get_active_flow_type()}_{st.session_state.flow_manager.get_flow_step()}_{i}_{st.session_state.button_counter}"
                        if st.button(option, key=button_key):
                            st.session_state.button_counter += 1
                            st.session_state.conversation_history.append({"role": "user", "content": option})
                            st.rerun()
            else:
                st.markdown(f"""
                <div class='chat-message assistant'>
                    <b>Noww Club AI</b><br>{msg['content']}
                </div>
                """, unsafe_allow_html=True)

# Add input form
st.markdown('<div class="input-container">', unsafe_allow_html=True)
with st.form(key="message_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Your message",
            placeholder="Ask questions, share thoughts, or request web searches...",
            label_visibility="collapsed",
            key="user_message_input"
        )
    with col2:
        send_button = st.form_submit_button("Send 💬", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Handle form submission
if send_button and user_input:
    try:
        # Create event loop if it doesn't exist
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async function
        response = loop.run_until_complete(process_user_input(user_input, st.session_state.flow_manager))
        st.rerun()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.rerun()

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><small>Built with ❤️ by Noww Club</small></p>
</div>
""", unsafe_allow_html=True)