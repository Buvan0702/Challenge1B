#!/usr/bin/env python3
"""
Adobe Hackathon Challenge 1B: Persona-Driven PDF Analysis
Advanced content analysis with relevance ranking and multi-collection processing.
"""

import os
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from datetime import datetime
import re
from collections import Counter
import math

# Import Challenge 1A extractor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Challenge_1a'))
from process_pdfs import PDFOutlineExtractor

import fitz  # PyMuPDF

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TFIDFAnalyzer:
    """TF-IDF based text analysis for relevance ranking."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.documents = []
        self.vocabulary = set()
        self.idf_scores = {}
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for TF-IDF analysis."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and split into words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 
            'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 
            'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 
            'too', 'use', 'that', 'with', 'have', 'this', 'will', 'your', 'from', 'they', 'know',
            'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 
            'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 
            'well', 'were', 'what'
        }
        
        return [word for word in words if word not in stop_words]
    
    def build_vocabulary(self, documents: List[str]):
        """Build vocabulary from all documents."""
        self.documents = []
        for doc in documents:
            processed = self.preprocess_text(doc)
            self.documents.append(processed)
            self.vocabulary.update(processed)
        
        # Calculate IDF scores
        num_docs = len(self.documents)
        for term in self.vocabulary:
            # Count documents containing this term
            doc_count = sum(1 for doc in self.documents if term in doc)
            # Calculate IDF
            self.idf_scores[term] = math.log(num_docs / (doc_count + 1))
    
    def calculate_tfidf_vector(self, text: str) -> Dict[str, float]:
        """Calculate TF-IDF vector for given text."""
        words = self.preprocess_text(text)
        word_counts = Counter(words)
        total_words = len(words)
        
        tfidf_vector = {}
        for term in self.vocabulary:
            # Calculate TF (term frequency)
            tf = word_counts.get(term, 0) / total_words if total_words > 0 else 0
            
            # Calculate TF-IDF
            tfidf_vector[term] = tf * self.idf_scores.get(term, 0)
        
        return tfidf_vector
    
    def cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two TF-IDF vectors."""
        # Get common terms
        common_terms = set(vec1.keys()) & set(vec2.keys())
        
        if not common_terms:
            return 0.0
        
        # Calculate dot product
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        
        # Calculate magnitudes
        mag1 = math.sqrt(sum(vec1[term] ** 2 for term in vec1))
        mag2 = math.sqrt(sum(vec2[term] ** 2 for term in vec2))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)


class PersonaDrivenAnalyzer:
    """Main analyzer for persona-driven content analysis."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.pdf_extractor = PDFOutlineExtractor()
        self.tfidf = TFIDFAnalyzer()
    
    def extract_pdf_content(self, pdf_path: str) -> Dict[str, Any]:
        """Extract full content from PDF including text blocks."""
        try:
            doc = fitz.open(pdf_path)
            content = {
                "title": self.pdf_extractor.extract_title(doc),
                "outline": [],
                "sections": [],
                "full_text": ""
            }
            
            full_text_parts = []
            current_section = None
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                full_text_parts.append(page_text)
                
                # Get structured text with font information
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            line_text = ""
                            line_fonts = []
                            
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    line_text += text + " "
                                    line_fonts.append({
                                        "size": span["size"],
                                        "flags": span["flags"]
                                    })
                            
                            if line_text.strip():
                                # Check if this could be a section header
                                avg_font_size = sum(f["size"] for f in line_fonts) / len(line_fonts) if line_fonts else 12
                                is_heading = any(f["size"] > 13 or f["flags"] & 16 for f in line_fonts)  # Bold or large
                                
                                if is_heading and len(line_text.strip()) < 150:
                                    # Start new section
                                    if current_section:
                                        content["sections"].append(current_section)
                                    
                                    current_section = {
                                        "title": line_text.strip(),
                                        "page": page_num + 1,
                                        "content": "",
                                        "subsections": []
                                    }
                                elif current_section:
                                    # Add to current section
                                    current_section["content"] += line_text
                                    
                                    # Check for subsections (smaller headings)
                                    if (any(f["flags"] & 16 for f in line_fonts) and 
                                        len(line_text.strip()) < 100 and
                                        len(line_text.strip()) > 10):
                                        current_section["subsections"].append({
                                            "title": line_text.strip(),
                                            "page": page_num + 1,
                                            "content": ""
                                        })
            
            # Add final section
            if current_section:
                content["sections"].append(current_section)
            
            content["full_text"] = "\n".join(full_text_parts)
            doc.close()
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting content from {pdf_path}: {e}")
            return {
                "title": f"Error processing {Path(pdf_path).name}",
                "outline": [],
                "sections": [],
                "full_text": ""
            }
    
    def analyze_collection(self, input_json_path: str, pdfs_dir: str) -> Dict[str, Any]:
        """Analyze a collection of PDFs based on persona and job requirements."""
        start_time = time.time()
        
        # Load input configuration
        with open(input_json_path, 'r') as f:
            input_config = json.load(f)
        
        persona = input_config["persona"]["role"]
        job_task = input_config["job_to_be_done"]["task"]
        documents = input_config["documents"]
        
        logger.info(f"Analyzing collection for persona: {persona}, task: {job_task}")
        
        # Extract content from all PDFs
        pdf_contents = {}
        all_text_content = []
        
        for doc_info in documents:
            pdf_path = os.path.join(pdfs_dir, doc_info["filename"])
            if os.path.exists(pdf_path):
                content = self.extract_pdf_content(pdf_path)
                pdf_contents[doc_info["filename"]] = content
                
                # Collect all text for TF-IDF analysis
                document_text = content["full_text"]
                for section in content["sections"]:
                    document_text += " " + section["title"] + " " + section["content"]
                
                all_text_content.append(document_text)
        
        # Build TF-IDF model
        self.tfidf.build_vocabulary(all_text_content)
        
        # Create query vector from persona and task
        query_text = f"{persona} {job_task}"
        query_vector = self.tfidf.calculate_tfidf_vector(query_text)
        
        # Rank sections by relevance
        section_rankings = []
        subsection_analysis = []
        
        for filename, content in pdf_contents.items():
            for section in content["sections"]:
                # Calculate relevance score
                section_text = f"{section['title']} {section['content']}"
                section_vector = self.tfidf.calculate_tfidf_vector(section_text)
                relevance_score = self.tfidf.cosine_similarity(query_vector, section_vector)
                
                section_rankings.append({
                    "document": filename,
                    "section_title": section["title"],
                    "page_number": section["page"],
                    "relevance_score": relevance_score,
                    "content": section["content"]
                })
                
                # Add subsection analysis for top sections
                if relevance_score > 0.1:  # Threshold for relevance
                    if section["content"].strip():
                        subsection_analysis.append({
                            "document": filename,
                            "refined_text": section["content"].strip()[:1000],  # Limit length
                            "page_number": section["page"]
                        })
        
        # Sort by relevance and assign ranks
        section_rankings.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Take top 5 sections
        extracted_sections = []
        for i, section in enumerate(section_rankings[:5]):
            extracted_sections.append({
                "document": section["document"],
                "section_title": section["section_title"],
                "importance_rank": i + 1,
                "page_number": section["page_number"]
            })
        
        # Take top 5 subsections
        subsection_analysis = subsection_analysis[:5]
        
        # Create output
        result = {
            "metadata": {
                "input_documents": [doc["filename"] for doc in documents],
                "persona": persona,
                "job_to_be_done": job_task,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }
        
        processing_time = time.time() - start_time
        logger.info(f"Completed analysis in {processing_time:.2f}s")
        
        return result


def process_collection(collection_dir: str):
    """Process a single collection directory."""
    input_file = os.path.join(collection_dir, "challenge1b_input.json")
    pdfs_dir = os.path.join(collection_dir, "PDFs")
    output_file = os.path.join(collection_dir, "challenge1b_output.json")
    
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return
    
    if not os.path.exists(pdfs_dir):
        logger.error(f"PDFs directory not found: {pdfs_dir}")
        return
    
    # Initialize analyzer
    analyzer = PersonaDrivenAnalyzer()
    
    # Analyze collection
    result = analyzer.analyze_collection(input_file, pdfs_dir)
    
    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Saved results to {output_file}")


def main():
    """Main function for Challenge 1B processing."""
    logger.info("Starting Challenge 1B: Persona-Driven PDF Analysis")
    
    # Get collections directory
    collections_dir = Path("/app/collections")
    if not collections_dir.exists():
        # Fall back to current directory for local testing
        collections_dir = Path(".")
    
    # Process each collection
    total_start_time = time.time()
    
    for collection_path in collections_dir.iterdir():
        if collection_path.is_dir() and collection_path.name.startswith("Collection"):
            logger.info(f"Processing {collection_path.name}")
            process_collection(str(collection_path))
    
    total_time = time.time() - total_start_time
    logger.info(f"Completed all collections in {total_time:.2f}s")


if __name__ == "__main__":
    main()
