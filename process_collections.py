import fitz  # PyMuPDF
import json
import os
import sys
import re
import traceback
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import spacy

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
except:
    pass

# Import key classes from Challenge 1A for PDF processing
@dataclass
class SpanProperties:
    """Data class for span formatting properties"""
    font_size: float
    font_family: str
    color: int
    is_bold: bool
    is_italic: bool
    flags: int

@dataclass
class BoundingBox:
    """Data class for bounding box coordinates"""
    x: float
    y: float
    width: float
    height: float

@dataclass
class BulletInfo:
    """Data class for bullet point information"""
    is_bullet: bool
    type: str = ""
    marker: str = ""
    text: str = ""
    level: int = 1

class TextAnalyzer:
    """Handles text pattern analysis and classification - from Challenge 1A"""
    
    BULLET_PATTERNS = {
        'numeric': [
            r'^(\d+)\.?\s+(.+)$',
            r'^(\d+)\)\s+(.+)$',
            r'^\((\d+)\)\s+(.+)$',
        ],
        'alpha_lower': [
            r'^([a-z])\.?\s+(.+)$',
            r'^([a-z])\)\s+(.+)$',
            r'^\(([a-z])\)\s+(.+)$',
        ],
        'alpha_upper': [
            r'^([A-Z])\.?\s+(.+)$',
            r'^([A-Z])\)\s+(.+)$',
            r'^\(([A-Z])\)\s+(.+)$',
        ],
        'symbol': [
            r'^[-•·▪▫◦‣⁃]\s+(.+)$',
            r'^[*]\s+(.+)$',
            r'^[→►]\s+(.+)$',
        ]
    }
    
    @classmethod
    def detect_bullet_patterns(cls, text: str) -> BulletInfo:
        """Detect various bullet point patterns"""
        text = text.strip()
        
        for category, patterns in cls.BULLET_PATTERNS.items():
            for pattern in patterns:
                match = re.match(pattern, text)
                if match:
                    if category == 'symbol':
                        return BulletInfo(
                            is_bullet=True,
                            type=category,
                            marker=match.group(0).split()[0],
                            text=match.group(1),
                            level=1
                        )
                    else:
                        marker = match.group(1)
                        content = match.group(2)
                        return BulletInfo(
                            is_bullet=True,
                            type=category,
                            marker=marker,
                            text=content,
                            level=1
                        )
        
        return BulletInfo(is_bullet=False)

class LayoutAnalyzer:
    """Handles document layout analysis and positioning - from Challenge 1A"""
    
    def __init__(self):
        self.page_margins = []
    
    def analyze_document_layout(self, doc) -> List[Dict]:
        """Analyze document layout to understand column structure and indentation patterns"""
        page_margins = []
        
        for page in doc:
            try:
                data = page.get_text("dict")
                left_margins = []
                
                for block in data["blocks"]:
                    if block["type"] != 0:
                        continue
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["text"].strip():
                                left_margins.append(span["origin"][0])
                
                if left_margins:
                    left_margins.sort()
                    margin_analysis = {
                        'min_margin': min(left_margins),
                        'common_margins': [],
                        'page_width': page.rect.width
                    }
                    
                    margin_clusters = self._cluster_margins(left_margins)
                    
                    for cluster in margin_clusters:
                        if len(cluster) >= 3:
                            margin_analysis['common_margins'].append(sum(cluster) / len(cluster))
                    
                    page_margins.append(margin_analysis)
                else:
                    page_margins.append({
                        'min_margin': 72,
                        'common_margins': [72],
                        'page_width': 612
                    })
            except Exception as e:
                page_margins.append({
                    'min_margin': 72,
                    'common_margins': [72],
                    'page_width': 612
                })
        
        self.page_margins = page_margins
        return page_margins
    
    def _cluster_margins(self, margins: List[float], tolerance: float = 20) -> List[List[float]]:
        """Cluster similar margin values"""
        if not margins:
            return []
            
        clusters = []
        current_cluster = [margins[0]]
        
        for margin in margins[1:]:
            if margin - current_cluster[-1] < tolerance:
                current_cluster.append(margin)
            else:
                clusters.append(current_cluster)
                current_cluster = [margin]
        clusters.append(current_cluster)
        
        return clusters

@dataclass
class DocumentSection:
    """Represents a document section with metadata"""
    document: str
    title: str
    content: str
    page_number: int
    font_size: float
    is_bold: bool
    position: Tuple[float, float]
    section_type: str
    indentation_level: int = 0
    relevance_score: float = 0.0
    topic_vector: Optional[np.ndarray] = None
    entities: List[str] = None

@dataclass
class SubSection:
    """Represents a refined subsection"""
    document: str
    content: str
    page_number: int
    relevance_score: float = 0.0
    topic_vector: Optional[np.ndarray] = None

class LightweightNLPProcessor:
    """Lightweight NLP processing using small models and efficient techniques"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.tfidf_vectorizer = None
        self.lsa_model = None
        self.stop_words = set()
        self.spacy_nlp = None
        self._initialize_nlp_components()
    
    def _initialize_nlp_components(self):
        """Initialize lightweight NLP components"""
        try:
            # Initialize stop words
            try:
                self.stop_words = set(stopwords.words('english'))
            except:
                # Fallback stop words if NLTK data not available
                self.stop_words = {
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                    'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                    'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
                }
            
            # Try to load small spaCy model (if available and under size limit)
            try:
                import spacy
                # Use small English model if available (~15MB)
                self.spacy_nlp = spacy.load("en_core_web_sm")
            except:
                self.spacy_nlp = None
                print("SpaCy model not available, using NLTK alternatives")
            
            # Initialize TF-IDF with memory-efficient settings
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,  # Limit vocabulary size
                stop_words='english',
                ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
                min_df=1,
                max_df=0.95,
                lowercase=True,
                strip_accents='ascii'
            )
            
            print("NLP components initialized successfully")
            
        except Exception as e:
            print(f"Warning: Some NLP components failed to initialize: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing"""
        if not text:
            return ""
        
        # Basic cleaning
        text = re.sub(r'[^\w\s\-\']', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization and stemming
        try:
            tokens = word_tokenize(text)
            # Filter out stop words and short tokens
            filtered_tokens = [
                self.stemmer.stem(token) for token in tokens 
                if len(token) > 2 and token not in self.stop_words and token.isalpha()
            ]
            return ' '.join(filtered_tokens)
        except:
            # Fallback if NLTK tokenization fails
            tokens = text.split()
            filtered_tokens = [
                token for token in tokens 
                if len(token) > 2 and token not in self.stop_words and token.isalpha()
            ]
            return ' '.join(filtered_tokens)
    
    def extract_named_entities(self, text: str) -> List[str]:
        """Extract named entities using available NLP tools"""
        entities = []
        
        if self.spacy_nlp:
            # Use spaCy if available
            try:
                doc = self.spacy_nlp(text)
                entities = [ent.text.lower() for ent in doc.ents 
                           if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']]
            except:
                pass
        
        if not entities:
            # Fallback to NLTK named entity recognition
            try:
                tokens = word_tokenize(text)
                pos_tags = pos_tag(tokens)
                named_entities = ne_chunk(pos_tags)
                
                for chunk in named_entities:
                    if hasattr(chunk, 'label'):
                        entity = ' '.join([token for token, pos in chunk.leaves()])
                        entities.append(entity.lower())
            except:
                pass
        
        return list(set(entities))  # Remove duplicates
    
    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[Tuple[str, float]]:
        """Extract key phrases using statistical methods"""
        if not text or len(text) < 50:
            return []
        
        try:
            # Use TF-IDF to identify important phrases
            sentences = sent_tokenize(text)
            if len(sentences) < 2:
                sentences = [text]
            
            # Create a temporary vectorizer for this text
            temp_vectorizer = TfidfVectorizer(
                max_features=500,
                stop_words='english',
                ngram_range=(2, 4),  # Focus on phrases
                min_df=1,
                lowercase=True
            )
            
            try:
                tfidf_matrix = temp_vectorizer.fit_transform(sentences)
                feature_names = temp_vectorizer.get_feature_names_out()
                
                # Get average TF-IDF scores
                mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                
                # Get top phrases
                phrase_scores = list(zip(feature_names, mean_scores))
                phrase_scores.sort(key=lambda x: x[1], reverse=True)
                
                return phrase_scores[:max_phrases]
            except:
                return []
                
        except Exception as e:
            return []

class TopicModelingEngine:
    """Lightweight topic modeling using LSA and clustering"""
    
    def __init__(self, nlp_processor: LightweightNLPProcessor):
        self.nlp_processor = nlp_processor
        self.tfidf_matrix = None
        self.lsa_model = None
        self.topic_vectors = None
        self.fitted = False
        
    def fit_topics(self, documents: List[str], n_topics: int = 20):
        """Fit topic model on document collection"""
        if not documents:
            return
        
        try:
            # Preprocess documents
            processed_docs = [self.nlp_processor.preprocess_text(doc) for doc in documents]
            processed_docs = [doc for doc in processed_docs if len(doc.strip()) > 10]
            
            if len(processed_docs) < 2:
                return
            
            # Create TF-IDF matrix
            self.nlp_processor.tfidf_vectorizer.fit(processed_docs)
            self.tfidf_matrix = self.nlp_processor.tfidf_vectorizer.transform(processed_docs)
            
            # Apply LSA for dimensionality reduction and topic discovery
            n_components = min(n_topics, min(len(processed_docs), self.tfidf_matrix.shape[1]) - 1)
            if n_components > 0:
                self.lsa_model = TruncatedSVD(n_components=n_components, random_state=42)
                self.topic_vectors = self.lsa_model.fit_transform(self.tfidf_matrix)
                self.fitted = True
                
                print(f"Topic model fitted with {n_components} topics")
            
        except Exception as e:
            print(f"Error fitting topic model: {e}")
    
    def get_document_topics(self, text: str) -> Optional[np.ndarray]:
        """Get topic vector for a document"""
        if not self.fitted or not text:
            return None
        
        try:
            processed_text = self.nlp_processor.preprocess_text(text)
            if len(processed_text.strip()) < 10:
                return None
            
            tfidf_vector = self.nlp_processor.tfidf_vectorizer.transform([processed_text])
            topic_vector = self.lsa_model.transform(tfidf_vector)
            return topic_vector.flatten()
            
        except Exception as e:
            print(f"Error getting document topics: {e}")
            return None
    
    def get_topic_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate cosine similarity between topic vectors"""
        try:
            if vector1 is None or vector2 is None:
                return 0.0
            
            # Ensure vectors are 2D for cosine_similarity
            v1 = vector1.reshape(1, -1) if vector1.ndim == 1 else vector1
            v2 = vector2.reshape(1, -1) if vector2.ndim == 1 else vector2
            
            similarity = cosine_similarity(v1, v2)[0, 0]
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

class AdvancedPersonaAnalyzer:
    """Advanced persona analysis using NLP techniques"""
    
    def __init__(self, nlp_processor: LightweightNLPProcessor):
        self.nlp_processor = nlp_processor
        self.persona_vector = None
        self.persona_keywords = set()
        self.persona_entities = set()
        self.task_phrases = []
    
    def analyze_persona(self, role: str, task: str) -> Dict[str, Any]:
        """Comprehensive persona analysis"""
        combined_text = f"{role} {task}"
        
        # Extract entities
        entities = self.nlp_processor.extract_named_entities(combined_text)
        self.persona_entities = set(entities)
        
        # Extract key phrases
        key_phrases = self.nlp_processor.extract_key_phrases(combined_text)
        self.task_phrases = [phrase for phrase, score in key_phrases if score > 0.1]
        
        # Process text for keywords
        processed_text = self.nlp_processor.preprocess_text(combined_text)
        self.persona_keywords = set(processed_text.split())
        
        # Create persona profile
        profile = {
            'role': role,
            'task': task,
            'processed_text': processed_text,
            'entities': list(self.persona_entities),
            'key_phrases': self.task_phrases,
            'keywords': list(self.persona_keywords),
            'combined_terms': list(self.persona_keywords) + self.task_phrases + entities
        }
        
        return profile

class SmartRelevanceScorer:
    """Advanced relevance scoring using multiple NLP techniques"""
    
    def __init__(self, persona_profile: Dict[str, Any], topic_engine: TopicModelingEngine):
        self.persona_profile = persona_profile
        self.topic_engine = topic_engine
        self.persona_terms = set(persona_profile.get('combined_terms', []))
        self.persona_vector = None
        
        # Create persona vector if topic model is available
        if topic_engine.fitted:
            persona_text = f"{persona_profile.get('role', '')} {persona_profile.get('task', '')}"
            self.persona_vector = topic_engine.get_document_topics(persona_text)
    
    def score_section(self, section: DocumentSection) -> float:
        """Comprehensive section scoring using multiple techniques"""
        total_score = 0.0
        
        # 1. Keyword-based scoring (40% weight)
        keyword_score = self._calculate_keyword_score(section)
        total_score += keyword_score * 0.4
        
        # 2. Topic similarity scoring (35% weight)
        topic_score = self._calculate_topic_score(section)
        total_score += topic_score * 0.35
        
        # 3. Entity-based scoring (15% weight)
        entity_score = self._calculate_entity_score(section)
        total_score += entity_score * 0.15
        
        # 4. Structural importance (10% weight)
        structural_score = self._calculate_structural_score(section)
        total_score += structural_score * 0.1
        
        return total_score
    
    def _calculate_keyword_score(self, section: DocumentSection) -> float:
        """Calculate score based on keyword overlap"""
        text = f"{section.title} {section.content}".lower()
        
        # Count overlapping terms
        text_words = set(text.split())
        overlap = len(text_words.intersection(self.persona_terms))
        
        if len(self.persona_terms) == 0:
            return 0.0
        
        # Calculate Jaccard similarity
        union_size = len(text_words.union(self.persona_terms))
        jaccard_score = overlap / union_size if union_size > 0 else 0.0
        
        # Weight by text length (avoid bias toward long texts)
        length_factor = 1.0 / (1.0 + math.log(len(text.split()) + 1))
        
        return jaccard_score * length_factor * 10.0  # Scale to reasonable range
    
    def _calculate_topic_score(self, section: DocumentSection) -> float:
        """Calculate score based on topic similarity"""
        if not self.topic_engine.fitted or self.persona_vector is None:
            return 0.0
        
        if section.topic_vector is None:
            section_text = f"{section.title} {section.content}"
            section.topic_vector = self.topic_engine.get_document_topics(section_text)
        
        if section.topic_vector is None:
            return 0.0
        
        similarity = self.topic_engine.get_topic_similarity(
            self.persona_vector, section.topic_vector
        )
        
        return similarity * 10.0  # Scale to reasonable range
    
    def _calculate_entity_score(self, section: DocumentSection) -> float:
        """Calculate score based on named entity overlap"""
        if not section.entities:
            return 0.0
        
        persona_entities = set(self.persona_profile.get('entities', []))
        section_entities = set(section.entities)
        
        if len(persona_entities) == 0:
            return 0.0
        
        overlap = len(section_entities.intersection(persona_entities))
        return (overlap / len(persona_entities)) * 10.0  # Scale to reasonable range
    
    def _calculate_structural_score(self, section: DocumentSection) -> float:
        """Calculate score based on structural importance"""
        score = 1.0
        
        # Section type importance
        type_weights = {
            'title': 3.0,
            'heading': 2.5,
            'subheading': 2.0,
            'paragraph': 1.0,
            'bullet_point': 1.5
        }
        score *= type_weights.get(section.section_type, 1.0)
        
        # Font size importance
        if section.font_size > 16:
            score *= 1.5
        elif section.font_size > 14:
            score *= 1.3
        elif section.font_size > 12:
            score *= 1.1
        
        # Bold text importance
        if section.is_bold:
            score *= 1.2
        
        return score

class IntelligentDocumentProcessor:
    """Advanced document processing using Challenge 1A components + NLP"""
    
    def __init__(self):
        self.text_analyzer = TextAnalyzer()
        self.layout_analyzer = LayoutAnalyzer()
        self.nlp_processor = LightweightNLPProcessor()
        
    def extract_document_content(self, pdf_path: str) -> Tuple[List[DocumentSection], Dict[str, Any]]:
        """Extract structured content with NLP enhancement"""
        try:
            doc = fitz.open(pdf_path)
            
            # Analyze document layout using Challenge 1A
            page_layouts = self.layout_analyzer.analyze_document_layout(doc)
            max_font_size = self._find_document_max_font_size(doc)
            
            sections = []
            filename = os.path.basename(pdf_path)
            
            for page_num, page in enumerate(doc):
                page_sections = self._extract_page_sections(
                    page, page_num, filename, max_font_size, 
                    page_layouts[page_num] if page_num < len(page_layouts) else {}
                )
                sections.extend(page_sections)
            
            doc.close()
            
            # Enhance sections with NLP analysis
            self._enhance_sections_with_nlp(sections)
            
            metadata = {
                'filename': filename,
                'total_pages': len(page_layouts),
                'sections_found': len(sections),
                'max_font_size': max_font_size
            }
            
            return sections, metadata
            
        except Exception as e:
            print(f"Error extracting from {pdf_path}: {e}")
            return [], {'filename': os.path.basename(pdf_path), 'error': str(e)}
    
    def _find_document_max_font_size(self, doc) -> float:
        """Find the maximum font size across the entire document"""
        max_font_size = 12
        try:
            for page in doc:
                data = page.get_text("dict")
                for block in data.get("blocks", []):
                    if block.get("type") != 0:
                        continue
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            if span.get("text", "").strip():
                                max_font_size = max(max_font_size, span.get("size", 12))
        except Exception as e:
            print(f"Error finding max font size: {e}")
        return max_font_size
    
    def _extract_page_sections(self, page, page_num: int, filename: str, 
                              max_font_size: float, page_margins: Dict) -> List[DocumentSection]:
        """Extract sections from a single page"""
        sections = []
        
        try:
            data = page.get_text("dict")
            
            for block in data.get("blocks", []):
                if block.get("type") != 0:
                    continue
                
                block_text = ""
                block_spans = []
                
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            block_text += text + " "
                            block_spans.append(span)
                
                if block_text.strip() and block_spans:
                    # Analyze block properties
                    first_span = block_spans[0]
                    font_sizes = [s.get("size", 12) for s in block_spans]
                    is_bold_flags = [(s.get("flags", 0) & 16) != 0 for s in block_spans]
                    
                    avg_font_size = sum(font_sizes) / len(font_sizes)
                    is_bold = sum(is_bold_flags) > len(is_bold_flags) / 2
                    
                    # Classify section type
                    section_type = self._classify_section_type(
                        block_text.strip(), avg_font_size, is_bold, 
                        max_font_size, page_num, first_span.get("origin", (0, 0))
                    )
                    
                    # Create section
                    section = DocumentSection(
                        document=filename,
                        title=block_text.strip()[:200] if section_type in ['heading', 'title'] else f"Section {len(sections)+1}",
                        content=block_text.strip(),
                        page_number=page_num + 1,
                        font_size=avg_font_size,
                        is_bold=is_bold,
                        position=first_span.get("origin", (0, 0)),
                        section_type=section_type,
                        indentation_level=0,
                        entities=[]
                    )
                    
                    sections.append(section)
                    
        except Exception as e:
            print(f"Error processing page {page_num} of {filename}: {e}")
        
        return sections
    
    def _classify_section_type(self, text: str, font_size: float, is_bold: bool, 
                              max_font_size: float, page_num: int, position: Tuple[float, float]) -> str:
        """Classify section type"""
        text_length = len(text)
        
        # Title detection
        if page_num == 0 and font_size >= max_font_size * 0.8 and text_length < 150:
            return "title"
        
        # Heading detection
        if (font_size > 14 or is_bold) and text_length < 200:
            return "heading"
        
        # Subheading detection
        if (font_size > 12 or is_bold) and text_length < 300:
            return "subheading"
        
        # Check for bullet points
        bullet_info = self.text_analyzer.detect_bullet_patterns(text)
        if bullet_info.is_bullet:
            return "bullet_point"
        
        return "paragraph"
    
    def _enhance_sections_with_nlp(self, sections: List[DocumentSection]):
        """Enhance sections with NLP analysis"""
        for section in sections:
            try:
                # Extract named entities
                section.entities = self.nlp_processor.extract_named_entities(
                    f"{section.title} {section.content}"
                )
            except Exception as e:
                section.entities = []

class DocumentIntelligenceSystem:
    """Main system with advanced NLP capabilities"""
    
    def __init__(self):
        self.nlp_processor = LightweightNLPProcessor()
        self.persona_analyzer = AdvancedPersonaAnalyzer(self.nlp_processor)
        self.document_processor = IntelligentDocumentProcessor()
        self.topic_engine = TopicModelingEngine(self.nlp_processor)
    
    def process_documents(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing pipeline with NLP integration"""
        try:
            # Extract input data
            documents = input_data.get('documents', [])
            persona_info = input_data.get('persona', {})
            job_info = input_data.get('job_to_be_done', {})
            
            # Analyze persona using NLP
            persona_profile = self.persona_analyzer.analyze_persona(
                persona_info.get('role', ''),
                job_info.get('task', '')
            )
            
            print(f"Analyzed persona: {len(persona_profile['keywords'])} keywords, {len(persona_profile['entities'])} entities")
            
            # Extract content from all documents
            all_sections = []
            document_metadata = []
            
            # Determine input directory based on input file location
            input_file_dir = os.path.dirname(input_data.get('_input_path', ''))
            
            possible_input_dirs = [
                "/app/input",
                "app/input",
                "app\\input\\Collection_1",
                "app/input/Collection_1",
                input_file_dir,
                os.path.join(input_file_dir, "PDFs") if input_file_dir else "PDFs",
                "input",
                ".",
                "PDFs"
            ]
            
            # Find the directory that contains the PDF files
            input_dir = None
            for potential_dir in possible_input_dirs:
                if potential_dir and os.path.exists(potential_dir):
                    # Check if this directory contains any of the required PDFs
                    pdf_count = 0
                    for doc_info in documents:
                        filename = doc_info.get('filename', '')
                        if os.path.exists(os.path.join(potential_dir, filename)):
                            pdf_count += 1
                    
                    if pdf_count > 0:
                        input_dir = potential_dir
                        break
            
            if input_dir is None:
                # Search recursively for PDF files
                print("Searching for PDF files recursively...")
                found_pdfs = {}
                for root, dirs, files in os.walk("."):
                    for file in files:
                        if file.endswith('.pdf'):
                            found_pdfs[file] = os.path.join(root, file)
                
                print(f"Found {len(found_pdfs)} PDF files:")
                for name, path in list(found_pdfs.items())[:10]:
                    print(f"  - {name} at {path}")
                
                # Use current directory as fallback
                input_dir = "."
            
            print(f"Looking for PDF files in: {input_dir}")
            
            for doc_info in documents:
                filename = doc_info.get('filename', '')
                pdf_path = os.path.join(input_dir, filename)
                
                if os.path.exists(pdf_path):
                    print(f"Processing {filename}...")
                    sections, metadata = self.document_processor.extract_document_content(pdf_path)
                    all_sections.extend(sections)
                    document_metadata.append(metadata)
                    print(f"Extracted {len(sections)} sections from {filename}")
                else:
                    print(f"Warning: File not found at {pdf_path}")
                    # Search for the file recursively
                    found = False
                    for root, dirs, files in os.walk("."):
                        if filename in files:
                            found_path = os.path.join(root, filename)
                            print(f"Found {filename} at: {found_path}")
                            sections, metadata = self.document_processor.extract_document_content(found_path)
                            all_sections.extend(sections)
                            document_metadata.append(metadata)
                            print(f"Extracted {len(sections)} sections from {filename}")
                            found = True
                            break
                    
                    if not found:
                        print(f"ERROR: Could not find {filename} anywhere!")
            
            print(f"Total sections extracted: {len(all_sections)}")
            
            # Fit topic model on all document content
            all_texts = [f"{s.title} {s.content}" for s in all_sections if s.content.strip()]
            if all_texts:
                self.topic_engine.fit_topics(all_texts, n_topics=min(20, len(all_texts)))
            
            # Generate topic vectors for sections
            for section in all_sections:
                section.topic_vector = self.topic_engine.get_document_topics(
                    f"{section.title} {section.content}"
                )
            
            # Initialize relevance scorer
            relevance_scorer = SmartRelevanceScorer(persona_profile, self.topic_engine)
            
            # Score all sections
            for section in all_sections:
                section.relevance_score = relevance_scorer.score_section(section)
            
            # Filter and sort by relevance
            relevant_sections = [s for s in all_sections if s.relevance_score > 0.5]
            relevant_sections.sort(key=lambda x: x.relevance_score, reverse=True)
            
            print(f"Relevant sections after NLP filtering: {len(relevant_sections)}")
            
            # Get top sections
            top_sections = relevant_sections[:5]
            
            # Extract intelligent subsections
            subsections = self._extract_intelligent_subsections(
                top_sections, relevance_scorer
            )
            
            print(f"Generated {len(subsections)} intelligent subsections")
            
            # Create output
            output = self._create_output(input_data, top_sections, subsections, document_metadata)
            
            return output
            
        except Exception as e:
            print(f"Error in processing: {e}")
            traceback.print_exc()
            return self._create_error_output(str(e))
    
    def _extract_intelligent_subsections(self, sections: List[DocumentSection], 
                                       scorer: SmartRelevanceScorer, 
                                       max_subsections: int = 5) -> List[SubSection]:
        """Extract subsections using NLP techniques"""
        subsections = []
        
        for section in sections[:max_subsections]:
            section_subsections = self._extract_from_section(section, scorer)
            subsections.extend(section_subsections)
        
        # Sort by relevance and return top ones
        subsections.sort(key=lambda x: x.relevance_score, reverse=True)
        return subsections[:max_subsections]
    
    def _extract_from_section(self, section: DocumentSection, 
                            scorer: SmartRelevanceScorer) -> List[SubSection]:
        """Extract meaningful subsections using NLP chunking"""
        subsections = []
        content = section.content
        
        if len(content) < 100:
            # Short sections use as-is
            subsection = SubSection(
                document=section.document,
                content=content,
                page_number=section.page_number,
                relevance_score=self._score_subsection_content(content, scorer)
            )
            subsections.append(subsection)
            return subsections
        
        # Use sentence-based chunking for better semantic coherence
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(content)
        except:
            # Fallback sentence splitting
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # Group sentences into meaningful chunks
        chunks = self._create_semantic_chunks(sentences)
        
        for chunk in chunks:
            if len(chunk.strip()) > 50:
                subsection = SubSection(
                    document=section.document,
                    content=chunk.strip(),
                    page_number=section.page_number,
                    relevance_score=self._score_subsection_content(chunk, scorer)
                )
                subsections.append(subsection)
        
        return subsections
    
    def _create_semantic_chunks(self, sentences: List[str], 
                              target_length: int = 300) -> List[str]:
        """Create semantically coherent chunks from sentences"""
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Add sentence to current chunk if it fits
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= target_length:
                current_chunk = potential_chunk
            else:
                # Start new chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    def _score_subsection_content(self, content: str, scorer: SmartRelevanceScorer) -> float:
        """Score subsection content using NLP techniques"""
        # Create temporary section for scoring
        temp_section = DocumentSection(
            document="temp",
            title="",
            content=content,
            page_number=1,
            font_size=12,
            is_bold=False,
            position=(0, 0),
            section_type="paragraph"
        )
        
        # Generate topic vector for subsection
        temp_section.topic_vector = self.topic_engine.get_document_topics(content)
        temp_section.entities = self.nlp_processor.extract_named_entities(content)
        
        return scorer.score_section(temp_section)
    
    def _create_output(self, input_data: Dict, sections: List[DocumentSection], 
                      subsections: List[SubSection], metadata: List[Dict]) -> Dict[str, Any]:
        """Create the final output JSON in the exact required format"""
        
        output = {
            "metadata": {
                "input_documents": [doc.get('filename', '') for doc in input_data.get('documents', [])],
                "persona": input_data.get('persona', {}).get('role', ''),
                "job_to_be_done": input_data.get('job_to_be_done', {}).get('task', ''),
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }
        
        # Add top sections in exact required format
        for i, section in enumerate(sections, 1):
            # Clean section title - use actual title for headings, content preview for others
            if section.section_type in ['heading', 'title', 'subheading']:
                section_title = section.title
            else:
                # For paragraph content, create a descriptive title from content
                content_words = section.content.split()[:8]  # First 8 words
                section_title = ' '.join(content_words)
                if len(section.content.split()) > 8:
                    section_title += "..."
                # Capitalize first letter
                section_title = section_title[0].upper() + section_title[1:] if section_title else "Content"
            
            output["extracted_sections"].append({
                "document": section.document,
                "section_title": section_title,
                "importance_rank": i,
                "page_number": section.page_number
            })
        
        # Add subsections in exact required format
        for subsection in subsections:
            output["subsection_analysis"].append({
                "document": subsection.document,
                "refined_text": subsection.content,
                "page_number": subsection.page_number
            })
        
        return output
    
    def _create_error_output(self, error_msg: str) -> Dict[str, Any]:
        """Create error output"""
        return {
            "metadata": {
                "error": error_msg,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }

def main():
    """Main function"""
    try:
        # Read input - try multiple possible locations
        possible_input_paths = [
            "/app/input/challenge1b_input.json",
            "challenge1b_input.json",
            "input/challenge1b_input.json",
            "app/input/challenge1b_input.json",
            "app\\input\\Collection_1\\challenge1b_input.json",
            "app/input/Collection_1/challenge1b_input.json",
            "Collection_1/challenge1b_input.json",
            os.path.join("app", "input", "Collection_1", "challenge1b_input.json")
        ]
        
        # Also search recursively for any input.json files
        for root, dirs, files in os.walk("."):
            for file in files:
                if "input.json" in file.lower() or "challenge1b_input.json" in file.lower():
                    possible_input_paths.append(os.path.join(root, file))
        
        input_path = None
        for path in possible_input_paths:
            if os.path.exists(path):
                input_path = path
                break
        
        if input_path is None:
            print("Input file not found in any of the expected locations:")
            for path in possible_input_paths[:8]:  # Show first 8 attempted paths
                print(f"  - {path}")
            print("\nSearching for input files...")
            
            # Show what files we can find
            found_files = []
            for root, dirs, files in os.walk("."):
                for file in files:
                    if file.endswith('.json'):
                        found_files.append(os.path.join(root, file))
            
            if found_files:
                print("Found these JSON files:")
                for f in found_files:
                    print(f"  - {f}")
                print("\nPlease rename one of these to 'challenge1b_input.json' or place it in the expected location.")
            
            sys.exit(1)
        
        print(f"Found input file: {input_path}")
        
        # Determine output path and directory based on input location
        input_dir = os.path.dirname(input_path)
        if "Collection_1" in input_path:
            output_dir = input_dir.replace("input", "output")
            output_path = os.path.join(output_dir, "challenge1b_output.json")
        elif input_path.startswith("/app/"):
            output_dir = "/app/output"
            output_path = "/app/output/challenge1b_output.json"
        else:
            # For local testing
            if input_dir:
                output_dir = input_dir.replace("input", "output") if "input" in input_dir else input_dir
            else:
                output_dir = "output"
            output_path = os.path.join(output_dir, "challenge1b_output.json")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # Add input path to data for reference
        input_data['_input_path'] = input_path
        
        print("="*60)
        print("Advanced NLP Document Intelligence System")
        print("="*60)
        print(f"Processing {len(input_data.get('documents', []))} documents...")
        print(f"Persona: {input_data.get('persona', {}).get('role', 'Unknown')}")
        print(f"Task: {input_data.get('job_to_be_done', {}).get('task', 'Unknown')}")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print("="*60)
        
        # Initialize system and process
        system = DocumentIntelligenceSystem()
        result = system.process_documents(input_data)
        
        # Ensure output directory exists
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save output
        print(f"Saving results to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print("="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Output saved to: {output_path}")
        print(f"Relevant sections found: {len(result.get('extracted_sections', []))}")
        print(f"Intelligent subsections generated: {len(result.get('subsection_analysis', []))}")
        
        # Print summary of top results
        if result.get('extracted_sections'):
            print("\nTop Extracted Sections:")
            for section in result['extracted_sections'][:3]:
                print(f"  {section['importance_rank']}. {section['section_title'][:80]}...")
                print(f"     Document: {section['document']}, Page: {section['page_number']}")
        
        if result.get('subsection_analysis'):
            print(f"\nTop Subsection Preview:")
            top_subsection = result['subsection_analysis'][0]
            preview = top_subsection['refined_text'][:150] + "..."
            print(f"  {preview}")
            print(f"  Document: {top_subsection['document']}, Page: {top_subsection['page_number']}")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()