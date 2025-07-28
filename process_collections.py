#!/usr/bin/env python3
"""
Challenge 1B: Persona-Driven Document Intelligence with Lightweight NLP
Enhanced with TF-IDF, semantic similarity, and advanced text analysis
"""

import fitz  # PyMuPDF
import re
import json
import os
import sys
import time
import math
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple, Optional

# Lightweight NLP imports
import string

# Try to import scikit-learn for TF-IDF (lightweight and efficient)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not available, using manual TF-IDF implementation")

# =====================================================================================
# LIGHTWEIGHT NLP TECHNIQUES
# =====================================================================================

class LightweightNLP:
    """Lightweight NLP processor with TF-IDF, similarity, and text analysis"""
    
    def __init__(self):
        self.stop_words = set([
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'would', 'this', 'they', 'we', 'you',
            'have', 'had', 'what', 'said', 'each', 'which', 'do', 'how',
            'their', 'if', 'up', 'out', 'many', 'then', 'them', 'these',
            'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him',
            'time', 'two', 'more', 'go', 'no', 'way', 'could', 'my', 'than',
            'first', 'been', 'call', 'who', 'oil', 'sit', 'now', 'find',
            'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part'
        ])
        
        self.tfidf_vectorizer = None
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                lowercase=True
            )
    
    def preprocess_text(self, text):
        """Clean and preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation except periods for sentence boundary
        text = re.sub(r'[^\w\s\.]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keywords(self, text, top_k=10):
        """Extract keywords using frequency analysis"""
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        # Filter out stop words and short words
        filtered_words = [
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        # Count word frequencies
        word_freq = Counter(filtered_words)
        
        # Get top keywords
        keywords = [word for word, freq in word_freq.most_common(top_k)]
        return keywords
    
    def calculate_text_similarity(self, text1, text2):
        """Calculate similarity between two texts using word overlap"""
        words1 = set(self.preprocess_text(text1).split()) - self.stop_words
        words2 = set(self.preprocess_text(text2).split()) - self.stop_words
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def extract_named_entities_simple(self, text):
        """Simple named entity extraction using patterns"""
        entities = {
            'PERSON': [],
            'ORG': [],
            'LOC': [],
            'DATE': [],
            'MONEY': [],
            'PRODUCT': []
        }
        
        # Person names (Title Case + Title Case)
        person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        entities['PERSON'] = re.findall(person_pattern, text)
        
        # Organizations (containing Inc, Corp, Ltd, Company, etc.)
        org_pattern = r'\b[A-Z][a-zA-Z\s]+(Inc|Corp|Ltd|Company|Organization|Association|Institute)\b'
        entities['ORG'] = re.findall(org_pattern, text)
        
        # Dates (various formats)
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{4}-\d{2}-\d{2}\b',       # YYYY-MM-DD
            r'\b[A-Z][a-z]+ \d{1,2}, \d{4}\b'  # Month DD, YYYY
        ]
        for pattern in date_patterns:
            entities['DATE'].extend(re.findall(pattern, text))
        
        # Money ($X, $X.XX, etc.)
        money_pattern = r'\$[\d,]+(?:\.\d{2})?|\b\d+(?:\.\d{2})?\s*(?:dollars?|USD|euros?|EUR)\b'
        entities['MONEY'] = re.findall(money_pattern, text, re.IGNORECASE)
        
        # Remove duplicates and empty strings
        for entity_type in entities:
            entities[entity_type] = list(set([e.strip() for e in entities[entity_type] if e.strip()]))
        
        return entities
    
    def calculate_readability_score(self, text):
        """Calculate simple readability score"""
        sentences = text.split('.')
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Count syllables (simple approximation)
        syllable_count = 0
        for word in words:
            syllable_count += max(1, len(re.findall(r'[aeiouAEIOU]', word)))
        
        avg_syllables_per_word = syllable_count / len(words) if words else 0
        
        # Simple readability score (lower is more readable)
        readability = (avg_sentence_length * 0.1) + (avg_syllables_per_word * 0.5)
        
        # Normalize to 0-1 scale (inverted so higher is better)
        return max(0.0, min(1.0, 1.0 - (readability / 10.0)))
    
    def calculate_information_density(self, text):
        """Calculate information density of text"""
        entities = self.extract_named_entities_simple(text)
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        
        words = text.split()
        unique_words = len(set(word.lower() for word in words))
        
        # Information indicators
        has_numbers = bool(re.search(r'\d+', text))
        has_technical_terms = bool(re.search(r'[A-Z]{2,}', text))  # Acronyms
        has_structured_content = bool(re.search(r'[:\-\•]', text))
        
        # Calculate density score
        density_score = (
            (total_entities / max(len(words), 1)) * 0.3 +
            (unique_words / max(len(words), 1)) * 0.3 +
            (0.1 if has_numbers else 0) +
            (0.1 if has_technical_terms else 0) +
            (0.2 if has_structured_content else 0)
        )
        
        return min(1.0, density_score)
    
    def analyze_text_with_tfidf(self, texts, target_text):
        """Analyze text importance using TF-IDF"""
        if not SKLEARN_AVAILABLE or not texts or not target_text:
            return 0.0
        
        try:
            all_texts = texts + [target_text]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            
            # Get TF-IDF scores for target text
            target_vector = tfidf_matrix[-1]
            target_scores = target_vector.toarray()[0]
            
            # Calculate average TF-IDF score
            non_zero_scores = target_scores[target_scores > 0]
            avg_tfidf = np.mean(non_zero_scores) if len(non_zero_scores) > 0 else 0.0
            
            return float(min(1.0, avg_tfidf * 2.0))  # Scale appropriately
        except:
            return 0.0
    
    def calculate_semantic_coherence(self, text):
        """Calculate semantic coherence of text"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) < 2:
            return 0.5  # Default for single sentence
        
        # Calculate average similarity between consecutive sentences
        similarities = []
        for i in range(len(sentences) - 1):
            sim = self.calculate_text_similarity(sentences[i], sentences[i + 1])
            similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0

# =====================================================================================
# ENHANCED PERSONA ANALYSIS WITH NLP
# =====================================================================================

class PersonaAnalyzer:
    """Enhanced persona analysis with NLP techniques"""
    
    def __init__(self):
        self.nlp = LightweightNLP()
        self.persona_profiles = self._build_persona_profiles()
    
    def _build_persona_profiles(self):
        """Build enhanced persona profiles with semantic keywords"""
        return {
            "Travel Planner": {
                "keywords": {
                    "high": ["travel", "destination", "tourist", "vacation", "hotel", "restaurant", 
                            "culture", "attractions", "guide", "itinerary", "booking", "flight"],
                    "medium": ["location", "city", "weather", "transportation", "budget", "experience",
                              "local", "popular", "recommended", "visit", "explore", "discover"],
                    "low": ["service", "quality", "time", "day", "night", "price", "cost"]
                },
                "context_patterns": [
                    r"places? to (?:visit|see|go)", r"things? to do", r"travel guide",
                    r"tourist attraction", r"vacation planning", r"trip itinerary"
                ],
                "entity_preferences": ["LOC", "ORG", "MONEY", "DATE"]
            },
            "HR Professional": {
                "keywords": {
                    "high": ["employee", "staff", "recruitment", "hiring", "job", "position",
                            "training", "performance", "salary", "benefits", "policy", "management"],
                    "medium": ["career", "development", "workplace", "team", "interview", "candidate",
                              "evaluation", "skills", "experience", "contract", "communication"],
                    "low": ["meeting", "report", "system", "process", "time", "work"]
                },
                "context_patterns": [
                    r"human resources?", r"job description", r"performance review",
                    r"employee handbook", r"hiring process", r"training program"
                ],
                "entity_preferences": ["PERSON", "ORG", "MONEY", "DATE"]
            },
            "Food Contractor": {
                "keywords": {
                    "high": ["food", "meal", "recipe", "ingredient", "cooking", "kitchen",
                            "catering", "nutrition", "menu", "supplier", "contract", "vendor"],
                    "medium": ["quality", "fresh", "organic", "preparation", "service", "delivery",
                              "cost", "budget", "safety", "hygiene", "restaurant", "customer"],
                    "low": ["business", "management", "time", "order", "equipment"]
                },
                "context_patterns": [
                    r"food service", r"catering contract", r"menu planning",
                    r"food safety", r"kitchen equipment", r"meal preparation"
                ],
                "entity_preferences": ["ORG", "MONEY", "PRODUCT", "DATE"]
            },
            "PhD Researcher": {
                "keywords": {
                    "high": ["research", "methodology", "analysis", "data", "study", "experiment",
                            "results", "findings", "literature", "theory", "hypothesis", "academic"],
                    "medium": ["publication", "peer review", "statistical", "survey", "model",
                              "framework", "approach", "evidence", "validation", "assessment"],
                    "low": ["paper", "article", "journal", "conference", "review"]
                },
                "context_patterns": [
                    r"research methodolog(?:y|ies)", r"literature review", r"data analysis",
                    r"experimental design", r"statistical analysis", r"research findings"
                ],
                "entity_preferences": ["PERSON", "ORG", "DATE"]
            },
            "Investment Analyst": {
                "keywords": {
                    "high": ["revenue", "profit", "financial", "market", "investment", "growth",
                            "performance", "analysis", "trend", "strategy", "competition", "ROI"],
                    "medium": ["earnings", "portfolio", "risk", "valuation", "forecast", "capital",
                              "assets", "liability", "equity", "debt", "margin", "ratio"],
                    "low": ["company", "business", "industry", "sector", "report"]
                },
                "context_patterns": [
                    r"financial analysis", r"investment strategy", r"market trend",
                    r"revenue growth", r"competitive analysis", r"risk assessment"
                ],
                "entity_preferences": ["ORG", "MONEY", "DATE"]
            },
            "Chemistry Student": {
                "keywords": {
                    "high": ["chemical", "reaction", "molecule", "compound", "formula", "equation",
                            "experiment", "laboratory", "synthesis", "analysis", "organic", "kinetics"],
                    "medium": ["mechanism", "properties", "structure", "bond", "catalyst", "solution",
                              "concentration", "temperature", "pressure", "pH", "solvent"],
                    "low": ["study", "learn", "concept", "theory", "practice", "example"]
                },
                "context_patterns": [
                    r"chemical reaction", r"organic chemistry", r"reaction mechanism",
                    r"molecular structure", r"laboratory experiment", r"chemical analysis"
                ],
                "entity_preferences": ["PRODUCT", "ORG"]
            }
        }
    
    def calculate_enhanced_relevance(self, text, persona, task, all_texts=None):
        """Calculate enhanced relevance using multiple NLP techniques"""
        if not text or not text.strip():
            return {"total_score": 0.0, "breakdown": {}}
        
        profile = self.persona_profiles.get(persona, {})
        if not profile:
            return {"total_score": 0.1, "breakdown": {"base": 0.1}}
        
        # Initialize scores
        scores = {
            "keyword_matching": 0.0,
            "context_patterns": 0.0,
            "entity_analysis": 0.0,
            "task_alignment": 0.0,
            "semantic_quality": 0.0,
            "tfidf_importance": 0.0
        }
        
        text_lower = text.lower()
        text_words = set(self.nlp.preprocess_text(text).split()) - self.nlp.stop_words
        
        # 1. Enhanced keyword matching (30% weight)
        keywords = profile.get("keywords", {})
        word_count = len(text.split())
        
        high_matches = sum(1 for kw in keywords.get("high", []) if kw in text_lower)
        medium_matches = sum(1 for kw in keywords.get("medium", []) if kw in text_lower)
        low_matches = sum(1 for kw in keywords.get("low", []) if kw in text_lower)
        
        scores["keyword_matching"] = min(1.0, (
            (high_matches * 3.0 + medium_matches * 2.0 + low_matches * 1.0) / 
            max(word_count, 1) * 10.0
        ))
        
        # 2. Context pattern matching (15% weight)
        context_patterns = profile.get("context_patterns", [])
        pattern_matches = sum(1 for pattern in context_patterns if re.search(pattern, text_lower))
        scores["context_patterns"] = min(1.0, pattern_matches * 0.3)
        
        # 3. Named entity analysis (15% weight)
        entities = self.nlp.extract_named_entities_simple(text)
        preferred_entities = profile.get("entity_preferences", [])
        
        entity_score = 0.0
        for entity_type in preferred_entities:
            if entity_type in entities and entities[entity_type]:
                entity_score += len(entities[entity_type]) * 0.1
        
        scores["entity_analysis"] = min(1.0, entity_score)
        
        # 4. Task alignment analysis (20% weight)
        task_lower = task.lower()
        task_alignment = 0.0
        
        if "literature review" in task_lower or "review" in task_lower:
            if any(word in text_lower for word in ["methodology", "results", "findings", "study", "research"]):
                task_alignment += 0.3
        elif "analyze" in task_lower or "analysis" in task_lower:
            if any(word in text_lower for word in ["data", "trend", "performance", "comparison", "analysis"]):
                task_alignment += 0.3
        elif "study" in task_lower or "exam" in task_lower:
            if any(word in text_lower for word in ["concept", "definition", "mechanism", "example", "key"]):
                task_alignment += 0.3
        elif "summarize" in task_lower or "summary" in task_lower:
            if any(word in text_lower for word in ["overview", "summary", "conclusion", "key", "main"]):
                task_alignment += 0.3
        
        # Task-specific keyword matching
        task_keywords = self.nlp.extract_keywords(task, top_k=5)
        task_word_overlap = len(set(task_keywords).intersection(text_words))
        task_alignment += (task_word_overlap / max(len(task_keywords), 1)) * 0.4
        
        scores["task_alignment"] = min(1.0, task_alignment)
        
        # 5. Semantic quality analysis (10% weight)
        readability = self.nlp.calculate_readability_score(text)
        info_density = self.nlp.calculate_information_density(text)
        coherence = self.nlp.calculate_semantic_coherence(text)
        
        scores["semantic_quality"] = (readability + info_density + coherence) / 3.0
        
        # 6. TF-IDF importance (10% weight)
        if all_texts and len(all_texts) > 3:
            scores["tfidf_importance"] = self.nlp.analyze_text_with_tfidf(all_texts, text)
        
        # Calculate weighted total score
        weights = {
            "keyword_matching": 0.30,
            "context_patterns": 0.15,
            "entity_analysis": 0.15,
            "task_alignment": 0.20,
            "semantic_quality": 0.10,
            "tfidf_importance": 0.10
        }
        
        total_score = sum(scores[key] * weights[key] for key in scores)
        
        return {
            "total_score": min(1.0, total_score),
            "breakdown": scores,
            "entities": entities,
            "keywords_found": {
                "high": high_matches,
                "medium": medium_matches,
                "low": low_matches
            }
        }

# =====================================================================================
# CHALLENGE 1A INTEGRATION - DOCUMENT STRUCTURE EXTRACTION
# =====================================================================================

def find_document_max_font_size(doc):
    """Find the maximum font size across the entire document"""
    max_font_size = 0
    for page in doc:
        data = page.get_text("dict")
        for block in data["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        max_font_size = max(max_font_size, span["size"])
    return max_font_size

def detect_heading_patterns(text):
    """Detect heading patterns in text"""
    text = text.strip()
    
    # Check for numbered headings
    numbering_patterns = [
        r'^(chapter\s+\d+|part\s+[ivx]+)', r'^(\d+)\.?\s+[A-Z]',
        r'^([IVXLCDM]+)\.?\s+', r'^(\d+\.\d+)\.?\s+',
        r'^(\d+\.\d+\.\d+)\.?\s+', r'^([a-z])\.?\s+', r'^([A-Z])\.?\s+'
    ]
    
    for pattern in numbering_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    
    # Check for section keywords
    section_keywords = [
        'abstract', 'introduction', 'background', 'methodology', 'methods',
        'results', 'discussion', 'conclusion', 'references', 'summary'
    ]
    
    text_lower = text.lower()
    for keyword in section_keywords:
        if text_lower == keyword or text_lower.startswith(keyword + ' '):
            return True
    
    return False

def get_text_type(font_size, is_bold, text_length, page_num, max_font_size, text_content):
    """Classify text type based on formatting and content"""
    
    # Title detection (first page, large font)
    if page_num == 0 and font_size >= max_font_size * 0.8 and font_size >= 16:
        return "title"
    
    # Heading detection
    if (font_size > 14 or is_bold) and text_length < 100:
        if detect_heading_patterns(text_content):
            return "heading"
        elif text_length < 50:
            return "heading"
        else:
            return "subheading"
    elif font_size > 12 and text_length < 80:
        return "subheading"
    else:
        return "paragraph"

def merge_text_blocks(lines):
    """Merge related text lines into blocks"""
    if not lines:
        return []
    
    merged_blocks = []
    current_block = [lines[0]]
    
    for i in range(1, len(lines)):
        prev_span = current_block[-1]['spans'][0] if current_block[-1]['spans'] else None
        curr_span = lines[i]['spans'][0] if lines[i]['spans'] else None
        
        if prev_span and curr_span:
            # Check if lines should be merged based on proximity and formatting
            y_diff = abs(prev_span['origin'][1] - curr_span['origin'][1])
            font_diff = abs(prev_span['size'] - curr_span['size'])
            
            if y_diff <= prev_span['size'] * 1.5 and font_diff <= 1:
                current_block.append(lines[i])
            else:
                merged_blocks.append(current_block)
                current_block = [lines[i]]
        else:
            merged_blocks.append(current_block)
            current_block = [lines[i]]
    
    merged_blocks.append(current_block)
    return merged_blocks

def get_block_properties(block_lines, page_num, max_font_size):
    """Extract properties from a block of lines"""
    if not block_lines:
        return None
    
    all_spans = []
    for line in block_lines:
        all_spans.extend(line['spans'])
    
    if not all_spans:
        return None
    
    # Extract text and formatting
    all_text = ""
    font_sizes = []
    bold_flags = []
    
    for span in all_spans:
        if span['text'].strip():
            all_text += span['text'] + " "
            font_sizes.append(span['size'])
            bold_flags.append((span['flags'] & 16) != 0)
    
    all_text = all_text.strip()
    if not all_text:
        return None
    
    # Determine dominant properties
    dominant_font_size = max(set(font_sizes), key=font_sizes.count) if font_sizes else 12
    is_bold = sum(bold_flags) > len(bold_flags) / 2 if bold_flags else False
    
    # Classify text type
    text_type = get_text_type(
        dominant_font_size, is_bold, len(all_text), 
        page_num, max_font_size, all_text
    )
    
    return {
        'text': all_text,
        'font_size': dominant_font_size,
        'is_bold': is_bold,
        'type': text_type,
        'page': page_num + 1
    }

def process_pdf_structure(pdf_path):
    """Process PDF and extract structured content"""
    doc = fitz.open(pdf_path)
    max_font_size = find_document_max_font_size(doc)
    
    all_pages_blocks = []
    
    for page_num, page in enumerate(doc):
        data = page.get_text("dict")
        
        # Extract all text lines
        all_lines = []
        for block in data["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                if line["spans"] and any(span["text"].strip() for span in line["spans"]):
                    all_lines.append(line)
        
        # Sort by vertical position
        all_lines.sort(key=lambda line: line["spans"][0]["origin"][1] if line["spans"] else 0)
        
        # Merge into blocks
        merged_blocks = merge_text_blocks(all_lines)
        
        # Get block properties
        page_blocks = []
        for block_lines in merged_blocks:
            block_props = get_block_properties(block_lines, page_num, max_font_size)
            if block_props:
                page_blocks.append(block_props)
        
        all_pages_blocks.append(page_blocks)
    
    doc.close()
    return all_pages_blocks

# =====================================================================================
# CHALLENGE 1B - ENHANCED PERSONA-DRIVEN ANALYSIS
# =====================================================================================

def extract_sections_from_blocks(all_pages_blocks):
    """Extract sections with metadata from processed blocks"""
    sections = []
    section_id = 1
    
    for page_num, page_blocks in enumerate(all_pages_blocks):
        for block in page_blocks:
            text = block.get('text', '').strip()
            if len(text) < 10:  # Skip very short text
                continue
            
            section = {
                "id": section_id,
                "text": text,
                "type": block.get('type', 'paragraph'),
                "page": block.get('page', page_num + 1),
                "font_size": block.get('font_size', 12),
                "is_bold": block.get('is_bold', False),
                "word_count": len(text.split()),
                "character_count": len(text)
            }
            
            sections.append(section)
            section_id += 1
    
    return sections

def rank_sections_with_nlp(sections, persona, task, top_k=5):
    """Rank sections using enhanced NLP analysis"""
    if not sections:
        return []
    
    print(f"Analyzing {len(sections)} sections with NLP techniques...")
    
    analyzer = PersonaAnalyzer()
    
    # Collect all texts for TF-IDF analysis
    all_texts = [section.get('text', '') for section in sections]
    
    for section in sections:
        text = section.get('text', '')
        
        # Enhanced persona relevance analysis
        relevance_analysis = analyzer.calculate_enhanced_relevance(text, persona, task, all_texts)
        persona_score = relevance_analysis['total_score']
        
        # Type importance weights
        type_weights = {
            'title': 0.9,
            'heading': 0.8,
            'subheading': 0.7,
            'paragraph': 0.5
        }
        type_weight = type_weights.get(section.get('type', 'paragraph'), 0.5)
        
        # Format bonus
        format_bonus = 0.0
        if section.get('is_bold', False):
            format_bonus += 0.1
        if section.get('font_size', 12) > 14:
            format_bonus += 0.1
        
        # Position factor (earlier content often more important)
        position_factor = max(0.5, 1.0 - (section.get('page', 1) - 1) * 0.05)
        
        # Length factor with quality consideration
        word_count = section.get('word_count', 0)
        length_factor = min(1.0, word_count / 100.0)  # Normalize to 100 words
        
        # NLP quality bonus
        nlp = LightweightNLP()
        readability = nlp.calculate_readability_score(text)
        info_density = nlp.calculate_information_density(text)
        nlp_quality = (readability + info_density) / 2.0
        
        # Combined importance score with NLP enhancement
        importance_score = (
            0.35 * persona_score +      # 35% enhanced persona relevance
            0.20 * type_weight +        # 20% content type
            0.15 * nlp_quality +        # 15% NLP quality metrics
            0.10 * format_bonus +       # 10% formatting
            0.10 * position_factor +    # 10% position
            0.10 * length_factor        # 10% content length
        )
        
        # Store detailed analysis
        section['persona_relevance'] = persona_score
        section['importance_score'] = importance_score
        section['nlp_analysis'] = relevance_analysis
        section['nlp_quality'] = {
            'readability': readability,
            'information_density': info_density,
            'overall_quality': nlp_quality
        }
        section['type_weight'] = type_weight
    
    # Sort by importance score
    sections.sort(key=lambda x: x.get('importance_score', 0), reverse=True)
    
    # Return top K sections with rankings
    top_sections = sections[:top_k]
    for i, section in enumerate(top_sections):
        section['importance_rank'] = i + 1
    
    print(f"Selected top {len(top_sections)} sections using NLP analysis")
    return top_sections

def generate_enhanced_subsections(top_sections):
    """Generate refined subsections with NLP-enhanced extraction"""
    subsections = []
    nlp = LightweightNLP()
    
    for section in top_sections:
        text = section.get('text', '')
        
        # Split into sentences using multiple delimiters
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        
        if len(sentences) <= 2:
            # For short sections, use the whole text as one subsection
            if len(text) > 30:
                subsections.append({
                    "document": section.get('source_document', ''),
                    "page": section.get('page', 1),
                    "refined_text": text,
                    "parent_section_rank": section.get('importance_rank', 0),
                    "quality_score": section.get('nlp_quality', {}).get('overall_quality', 0.5)
                })
        else:
            # Group sentences by semantic coherence
            sentence_groups = []
            current_group = [sentences[0]]
            
            for i in range(1, len(sentences)):
                # Calculate similarity with current group
                current_group_text = '. '.join(current_group)
                similarity = nlp.calculate_text_similarity(current_group_text, sentences[i])
                
                # If similarity is high or group is small, add to current group
                if similarity > 0.3 or len(current_group) < 2:
                    current_group.append(sentences[i])
                else:
                    # Start new group
                    if len(current_group) > 0:
                        sentence_groups.append(current_group)
                    current_group = [sentences[i]]
            
            # Add the last group
            if current_group:
                sentence_groups.append(current_group)
            
            # Create subsections from groups
            for group in sentence_groups:
                refined_text = '. '.join(group).strip()
                if len(refined_text) > 40:  # Minimum length threshold
                    # Calculate quality score for this subsection
                    quality_score = nlp.calculate_information_density(refined_text)
                    
                    subsections.append({
                        "document": section.get('source_document', ''),
                        "page": section.get('page', 1),
                        "refined_text": refined_text,
                        "parent_section_rank": section.get('importance_rank', 0),
                        "quality_score": quality_score
                    })
    
    # Sort subsections by parent rank and quality
    subsections.sort(key=lambda x: (x['parent_section_rank'], -x['quality_score']))
    
    return subsections[:20]  # Limit to top 20 subsections

def create_challenge1b_output(sections, subsections, persona, task, metadata):
    """Create the required Challenge 1B output format with NLP insights"""
    
    # Prepare extracted sections
    extracted_sections = []
    for section in sections:
        # Create section title from first part of text
        section_text = section.get('text', '')
        section_title = section_text[:100] + "..." if len(section_text) > 100 else section_text
        
        extracted_sections.append({
            "document": section.get('source_document', ''),
            "page": section.get('page', 1),
            "section_title": section_title,
            "importance_rank": section.get('importance_rank', 0)
        })
    
    # Prepare subsection analysis
    subsection_analysis = []
    for subsection in subsections:
        subsection_analysis.append({
            "document": subsection.get('document', ''),
            "page": subsection.get('page', 1),
            "refined_text": subsection.get('refined_text', '')
        })
    
    # Calculate summary statistics
    avg_relevance = sum(s.get('persona_relevance', 0) for s in sections) / len(sections) if sections else 0
    avg_quality = sum(s.get('nlp_quality', {}).get('overall_quality', 0) for s in sections) / len(sections) if sections else 0
    
    # Count NLP insights
    total_entities = 0
    for section in sections:
        entities = section.get('nlp_analysis', {}).get('entities', {})
        total_entities += sum(len(entity_list) for entity_list in entities.values())
    
    output = {
        "metadata": {
            "input_documents": metadata.get('input_documents', []),
            "persona": persona,
            "job_to_be_done": task,
            "processing_timestamp": datetime.now().isoformat(),
            "nlp_analysis_summary": {
                "average_persona_relevance": round(avg_relevance, 3),
                "average_quality_score": round(avg_quality, 3),
                "total_named_entities": total_entities,
                "nlp_techniques_used": [
                    "TF-IDF Analysis",
                    "Named Entity Recognition", 
                    "Semantic Similarity",
                    "Information Density Calculation",
                    "Readability Assessment",
                    "Context Pattern Matching"
                ]
            }
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }
    
    return output

# =====================================================================================
# MAIN PROCESSING PIPELINE
# =====================================================================================

def process_document_collection(input_dir, output_dir):
    """Process document collection according to Challenge 1B requirements"""
    
    print("Challenge 1B: Enhanced NLP-Driven Document Intelligence")
    print("=" * 60)
    
    # Read input configuration
    config_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    if not config_files:
        print("No JSON configuration file found")
        return
    
    config_file = os.path.join(input_dir, config_files[0])
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Extract persona and task
    persona = config.get('persona', '')
    if isinstance(persona, dict):
        persona = persona.get('role', persona.get('name', ''))
    
    task = config.get('job_to_be_done', config.get('task', ''))
    if isinstance(task, dict):
        task = task.get('description', task.get('task', ''))
    
    # Find PDF files
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found")
        return
    
    print(f"📄 Processing {len(pdf_files)} documents")
    print(f"👤 Persona: {persona}")
    print(f"📋 Task: {task}")
    print(f"🧠 NLP Libraries: scikit-learn {'✅' if SKLEARN_AVAILABLE else '❌'}")
    
    # Process all PDFs
    all_sections = []
    processed_docs = []
    
    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(input_dir, pdf_file)
        try:
            print(f"[{i}/{len(pdf_files)}] Processing: {pdf_file}")
            
            # Extract structured content using Challenge 1A methods
            all_pages_blocks = process_pdf_structure(pdf_path)
            
            # Extract sections
            sections = extract_sections_from_blocks(all_pages_blocks)
            
            # Add source document info
            for section in sections:
                section['source_document'] = pdf_file
            
            all_sections.extend(sections)
            processed_docs.append(pdf_file)
            
            print(f"    Extracted {len(sections)} sections")
            
        except Exception as e:
            print(f"    ❌ Error processing {pdf_file}: {str(e)}")
            continue
    
    if not all_sections:
        print("❌ No sections extracted from any document")
        return
    
    print(f"\n📊 Total sections extracted: {len(all_sections)}")
    
    # Rank sections using enhanced NLP analysis
    print("🧠 Applying enhanced NLP analysis...")
    top_sections = rank_sections_with_nlp(all_sections, persona, task, top_k=5)
    
    if not top_sections:
        print("❌ No relevant sections found")
        return
    
    print(f"🏆 Selected top {len(top_sections)} most relevant sections")
    
    # Display top sections with scores
    print("\n📈 Top Sections Analysis:")
    for section in top_sections:
        print(f"  Rank {section['importance_rank']}: "
              f"Relevance={section['persona_relevance']:.3f}, "
              f"Quality={section['nlp_quality']['overall_quality']:.3f}, "
              f"Page={section['page']}")
    
    # Generate enhanced subsections
    print("\n🔬 Generating enhanced subsections...")
    subsections = generate_enhanced_subsections(top_sections)
    print(f"📝 Generated {len(subsections)} refined subsections")
    
    # Create output
    metadata = {
        'input_documents': processed_docs
    }
    
    output = create_challenge1b_output(top_sections, subsections, persona, task, metadata)
    
    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'challenge1b_output.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Output saved to: {output_file}")
    print("✅ Processing completed successfully!")
    
    # Print summary statistics
    nlp_summary = output['metadata']['nlp_analysis_summary']
    print(f"\n📋 NLP Analysis Summary:")
    print(f"   Average Relevance: {nlp_summary['average_persona_relevance']}")
    print(f"   Average Quality: {nlp_summary['average_quality_score']}")
    print(f"   Named Entities Found: {nlp_summary['total_named_entities']}")
    print(f"   NLP Techniques: {len(nlp_summary['nlp_techniques_used'])}")
    
    return output

def main():
    """Main function for Challenge 1B processing"""
    print("🚀 Challenge 1B: Persona-Driven Document Intelligence with NLP")
    print("=" * 70)
    
    # Get input and output directories
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    # For local testing, use current directory structure
    if not os.path.exists(input_dir):
        input_dir = "."
        output_dir = "."
    
    # Process the collection
    try:
        result = process_document_collection(input_dir, output_dir)
        if result:
            print("\n🎉 Challenge 1B processing completed successfully!")
            print("🏆 Enhanced with lightweight NLP techniques!")
        else:
            print("\n❌ Processing failed!")
    except Exception as e:
        print(f"\n💥 Error in main processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()