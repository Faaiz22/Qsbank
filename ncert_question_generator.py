import streamlit as st
import PyPDF2
import io
import re
import random
from typing import List, Dict, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree
from collections import Counter, defaultdict
import spacy
from datetime import datetime
import json
import pandas as pd

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('chunkers/maxent_ne_chunker')
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')

# Load spaCy model (fallback to basic processing if not available)
@st.cache_resource
def load_nlp_model():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except:
        return None

class AdvancedNCERTQuestionGenerator:
    def __init__(self):
        download_nltk_data()
        self.nlp = load_nlp_model()
        self.stop_words = set(stopwords.words('english'))
        
        # Educational keywords that indicate important concepts
        self.educational_keywords = {
            'definition': ['is', 'are', 'means', 'refers to', 'defined as', 'called'],
            'process': ['process', 'method', 'procedure', 'steps', 'mechanism'],
            'cause_effect': ['because', 'due to', 'causes', 'leads to', 'results in', 'therefore'],
            'comparison': ['compared to', 'different from', 'similar to', 'unlike', 'whereas'],
            'importance': ['important', 'significant', 'crucial', 'essential', 'vital'],
            'examples': ['example', 'instance', 'such as', 'including', 'like']
        }
        
        # Question templates for different types
        self.question_templates = {
            'definition': [
                "What is {term}?",
                "Define {term}.",
                "What does {term} mean?",
                "How is {term} defined?"
            ],
            'function': [
                "What is the function of {term}?",
                "What does {term} do?",
                "How does {term} work?",
                "What is the role of {term}?"
            ],
            'process': [
                "Explain the process of {term}.",
                "How does {term} occur?",
                "What are the steps involved in {term}?",
                "Describe the mechanism of {term}."
            ],
            'cause_effect': [
                "What causes {term}?",
                "What are the effects of {term}?",
                "Why does {term} happen?",
                "What results from {term}?"
            ],
            'comparison': [
                "How is {term1} different from {term2}?",
                "Compare {term1} and {term2}.",
                "What are the similarities between {term1} and {term2}?",
                "Distinguish between {term1} and {term2}."
            ],
            'application': [
                "Give examples of {term}.",
                "Where is {term} used?",
                "What are the applications of {term}?",
                "How is {term} applied in real life?"
            ]
        }
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text content from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def clean_and_segment_text(self, text: str) -> Dict[str, List[str]]:
        """Clean text and segment into chapters/sections"""
        # Remove page numbers, headers, footers
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'\d+\s*(NCERT|Chapter|Unit)', '', text)
        text = re.sub(r'[^\w\s\.\?\!\,\;\:\-\(\)\[\]\"\'\/]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Segment into chapters/sections
        segments = {}
        
        # Try to identify chapters
        chapter_pattern = r'(Chapter\s+\d+|CHAPTER\s+\d+|Unit\s+\d+|UNIT\s+\d+)(.*?)(?=Chapter\s+\d+|CHAPTER\s+\d+|Unit\s+\d+|UNIT\s+\d+|$)'
        chapters = re.findall(chapter_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if chapters:
            for i, (title, content) in enumerate(chapters):
                segments[f"Chapter_{i+1}_{title.strip()}"] = self.extract_meaningful_sentences(content)
        else:
            # If no chapters found, segment by length
            sentences = sent_tokenize(text)
            chunk_size = 50
            for i in range(0, len(sentences), chunk_size):
                chunk_sentences = sentences[i:i+chunk_size]
                segments[f"Section_{i//chunk_size + 1}"] = self.extract_meaningful_sentences(' '.join(chunk_sentences))
        
        return segments
    
    def extract_meaningful_sentences(self, text: str) -> List[str]:
        """Extract sentences that are likely to contain important information"""
        sentences = sent_tokenize(text)
        meaningful_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Filter criteria for meaningful sentences
            if (len(sentence) >= 30 and  # Minimum length
                len(sentence) <= 300 and  # Maximum length
                not sentence.lower().startswith(('figure', 'table', 'diagram', 'image')) and
                not re.match(r'^\d+\.', sentence) and  # Not a numbered list
                sentence.count('.') <= 3 and  # Not too many abbreviations
                any(keyword in sentence.lower() for keyword_list in self.educational_keywords.values() 
                    for keyword in keyword_list)):
                
                meaningful_sentences.append(sentence)
        
        return meaningful_sentences
    
    def extract_key_terms_and_concepts(self, text: str) -> Dict[str, List[str]]:
        """Extract key terms, definitions, and concepts using advanced NLP"""
        terms_and_concepts = {
            'definitions': [],
            'processes': [],
            'important_terms': [],
            'named_entities': [],
            'numerical_facts': []
        }
        
        # Use spaCy if available, otherwise fallback to NLTK
        if self.nlp:
            doc = self.nlp(text)
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'LAW', 'LANGUAGE']:
                    terms_and_concepts['named_entities'].append(ent.text)
            
            # Extract noun phrases that might be important terms
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 4 and chunk.text.lower() not in self.stop_words:
                    terms_and_concepts['important_terms'].append(chunk.text)
        
        # NLTK-based extraction
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            # Extract definitions (sentences with "is", "are", "means", etc.)
            if any(def_word in sentence.lower() for def_word in self.educational_keywords['definition']):
                terms_and_concepts['definitions'].append(sentence)
            
            # Extract processes
            if any(proc_word in sentence.lower() for proc_word in self.educational_keywords['process']):
                terms_and_concepts['processes'].append(sentence)
            
            # Extract numerical facts
            numbers = re.findall(r'\b\d+(?:\.\d+)?\s*(?:percent|%|degrees?|meters?|years?|times?)\b', sentence.lower())
            if numbers:
                terms_and_concepts['numerical_facts'].append(sentence)
        
        # Extract important terms using POS tagging
        words = word_tokenize(text.lower())
        tagged_words = pos_tag(words)
        
        # Focus on nouns and proper nouns
        important_words = []
        for word, tag in tagged_words:
            if (tag.startswith('NN') and 
                word not in self.stop_words and 
                len(word) > 3 and
                word.isalpha()):
                important_words.append(word.title())
        
        # Get most frequent terms
        word_freq = Counter(important_words)
        terms_and_concepts['important_terms'].extend([term for term, freq in word_freq.most_common(30)])
        
        # Remove duplicates
        for key in terms_and_concepts:
            terms_and_concepts[key] = list(set(terms_and_concepts[key]))
        
        return terms_and_concepts
    
    def generate_fill_in_blanks(self, sentences: List[str], terms: List[str], num_questions: int = 10) -> List[Dict]:
        """Generate high-quality fill-in-the-blank questions"""
        questions = []
        used_sentences = set()
        
        for sentence in sentences:
            if len(questions) >= num_questions:
                break
            
            if sentence in used_sentences:
                continue
            
            # Find the best term to blank out
            sentence_words = word_tokenize(sentence.lower())
            candidate_terms = [term for term in terms if term.lower() in sentence.lower()]
            
            if not candidate_terms:
                continue
            
            # Choose the most significant term (longest, most specific)
            target_term = max(candidate_terms, key=len)
            
            # Create question by replacing the term
            question_text = sentence.replace(target_term, "______")
            
            # Generate plausible distractors
            distractors = self.generate_distractors(target_term, terms, 3)
            options = [target_term] + distractors
            random.shuffle(options)
            
            correct_answer = chr(65 + options.index(target_term))
            
            questions.append({
                'id': len(questions) + 1,
                'type': 'fill_in_blank',
                'question': question_text,
                'options': {
                    'A': options[0],
                    'B': options[1],
                    'C': options[2],
                    'D': options[3]
                },
                'correct_answer': correct_answer,
                'explanation': f"The correct answer is '{target_term}' as mentioned in the source text.",
                'difficulty': self.assess_difficulty(sentence, target_term),
                'source': sentence
            })
            
            used_sentences.add(sentence)
        
        return questions
    
    def generate_definition_questions(self, definitions: List[str], num_questions: int = 5) -> List[Dict]:
        """Generate definition-based questions"""
        questions = []
        
        for definition in definitions[:num_questions]:
            # Extract the term being defined
            words = word_tokenize(definition)
            tagged = pos_tag(words)
            
            # Look for subject of the sentence (what is being defined)
            subject_candidates = []
            for i, (word, tag) in enumerate(tagged):
                if tag.startswith('NN') and i < len(words) // 2:  # Likely to be early in sentence
                    subject_candidates.append(word.title())
            
            if subject_candidates:
                term = subject_candidates[0]
                
                # Create question
                question_templates = [
                    f"What is {term}?",
                    f"Define {term}.",
                    f"How would you explain {term}?"
                ]
                
                questions.append({
                    'id': len(questions) + 1,
                    'type': 'definition',
                    'question': random.choice(question_templates),
                    'expected_answer': definition,
                    'key_terms': subject_candidates,
                    'difficulty': 'Easy',
                    'source': definition
                })
        
        return questions
    
    def generate_process_questions(self, processes: List[str], num_questions: int = 5) -> List[Dict]:
        """Generate questions about processes and mechanisms"""
        questions = []
        
        for process in processes[:num_questions]:
            # Extract process keywords
            process_indicators = ['process', 'method', 'procedure', 'steps', 'mechanism', 'way']
            
            for indicator in process_indicators:
                if indicator in process.lower():
                    # Create process-based questions
                    question_templates = [
                        f"Explain the process mentioned in the following context.",
                        f"What are the steps involved in the process described?",
                        f"How does the mechanism work as described?"
                    ]
                    
                    questions.append({
                        'id': len(questions) + 1,
                        'type': 'process',
                        'question': random.choice(question_templates),
                        'context': process,
                        'expected_elements': self.extract_process_steps(process),
                        'difficulty': 'Medium',
                        'source': process
                    })
                    break
        
        return questions
    
    def generate_analytical_questions(self, sentences: List[str], num_questions: int = 5) -> List[Dict]:
        """Generate analytical and reasoning questions"""
        questions = []
        
        # Look for cause-effect relationships
        cause_effect_sentences = [s for s in sentences 
                                 if any(indicator in s.lower() 
                                       for indicator in self.educational_keywords['cause_effect'])]
        
        for sentence in cause_effect_sentences[:num_questions]:
            # Identify cause and effect
            if 'because' in sentence.lower():
                parts = sentence.lower().split('because')
                effect = parts[0].strip()
                cause = parts[1].strip()
            elif 'due to' in sentence.lower():
                parts = sentence.lower().split('due to')
                effect = parts[0].strip()
                cause = parts[1].strip()
            else:
                continue
            
            questions.append({
                'id': len(questions) + 1,
                'type': 'analytical',
                'question': f"What is the cause of the following effect: {effect}?",
                'expected_answer': cause,
                'analysis_type': 'cause_effect',
                'difficulty': 'Hard',
                'source': sentence
            })
        
        return questions
    
    def generate_distractors(self, correct_term: str, all_terms: List[str], num_distractors: int) -> List[str]:
        """Generate plausible wrong answers for MCQs"""
        # Filter terms that are different from correct term
        candidates = [term for term in all_terms 
                     if term != correct_term and 
                     len(term.split()) == len(correct_term.split())]
        
        # Prefer terms with similar characteristics
        if len(candidates) < num_distractors:
            candidates.extend([f"Not {correct_term}", "None of the above", "All of the above"])
        
        return random.sample(candidates, min(num_distractors, len(candidates)))
    
    def extract_process_steps(self, text: str) -> List[str]:
        """Extract steps from a process description"""
        steps = []
        
        # Look for numbered steps
        numbered_steps = re.findall(r'\d+[.)]\s*([^.]+)', text)
        if numbered_steps:
            steps.extend(numbered_steps)
        
        # Look for sequence indicators
        sequence_words = ['first', 'second', 'third', 'then', 'next', 'finally', 'lastly']
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            if any(seq_word in sentence.lower() for seq_word in sequence_words):
                steps.append(sentence.strip())
        
        return steps
    
    def assess_difficulty(self, sentence: str, term: str) -> str:
        """Assess the difficulty level of a question"""
        # Simple heuristics for difficulty assessment
        word_count = len(sentence.split())
        term_complexity = len(term.split())
        
        if word_count < 15 and term_complexity == 1:
            return "Easy"
        elif word_count > 25 or term_complexity > 2:
            return "Hard"
        else:
            return "Medium"
    
    def validate_questions(self, questions: List[Dict]) -> List[Dict]:
        """Validate and filter out low-quality questions"""
        validated_questions = []
        
        for question in questions:
            # Basic validation rules
            if (len(question.get('question', '')) > 10 and
                'source' in question and
                len(question['source']) > 20):
                
                # Additional validation based on question type
                if question['type'] == 'fill_in_blank':
                    if ('options' in question and 
                        len(question['options']) == 4 and
                        'correct_answer' in question):
                        validated_questions.append(question)
                else:
                    validated_questions.append(question)
        
        return validated_questions

def main():
    st.set_page_config(
        page_title="Advanced NCERT Question Generator",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 Advanced NCERT Question Generator")
    st.markdown("**Generate high-quality, accurate questions from NCERT PDFs using advanced NLP techniques**")
    
    # Initialize the question generator
    if 'generator' not in st.session_state:
        with st.spinner("Initializing advanced question generator..."):
            st.session_state.generator = AdvancedNCERTQuestionGenerator()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Question Configuration")
        
        st.subheader("Question Types & Quantities")
        num_fill_blank = st.slider("Fill-in-the-Blank Questions", 5, 25, 15)
        num_definitions = st.slider("Definition Questions", 3, 15, 8)
        num_processes = st.slider("Process Questions", 2, 10, 5)
        num_analytical = st.slider("Analytical Questions", 2, 10, 5)
        
        st.subheader("Quality Filters")
        min_sentence_length = st.slider("Minimum Sentence Length", 20, 100, 30)
        difficulty_filter = st.multiselect(
            "Include Difficulty Levels", 
            ["Easy", "Medium", "Hard"], 
            default=["Easy", "Medium", "Hard"]
        )
        
        st.subheader("Export Options")
        include_explanations = st.checkbox("Include Explanations", value=True)
        include_source = st.checkbox("Include Source Text", value=True)
        export_format = st.selectbox("Export Format", ["JSON", "Excel", "Text", "CSV"])
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📚 Upload NCERT PDF")
        uploaded_file = st.file_uploader(
            "Select your NCERT textbook PDF",
            type="pdf",
            help="Upload a high-quality NCERT PDF for best results"
        )
        
        if uploaded_file:
            st.success(f"✅ {uploaded_file.name}")
            st.info(f"📊 Size: {len(uploaded_file.getvalue())/1024/1024:.1f} MB")
    
    with col2:
        if uploaded_file:
            st.header("⚡ Processing Pipeline")
            
            # Processing steps
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Text Extraction
                status_text.info("🔍 Extracting text from PDF...")
                progress_bar.progress(10)
                text = st.session_state.generator.extract_text_from_pdf(uploaded_file)
                
                if not text:
                    st.error("❌ Failed to extract text from PDF")
                    st.stop()
                
                # Step 2: Text Segmentation
                status_text.info("📑 Analyzing document structure...")
                progress_bar.progress(25)
                segments = st.session_state.generator.clean_and_segment_text(text)
                
                # Step 3: Concept Extraction
                status_text.info("🧠 Extracting key concepts and terms...")
                progress_bar.progress(40)
                all_sentences = []
                for segment_sentences in segments.values():
                    all_sentences.extend(segment_sentences)
                
                concepts = st.session_state.generator.extract_key_terms_and_concepts(' '.join(all_sentences))
                
                # Step 4: Question Generation
                status_text.info("❓ Generating questions...")
                progress_bar.progress(60)
                
                # Generate different types of questions
                fill_blank_questions = st.session_state.generator.generate_fill_in_blanks(
                    all_sentences, concepts['important_terms'], num_fill_blank
                )
                
                definition_questions = st.session_state.generator.generate_definition_questions(
                    concepts['definitions'], num_definitions
                )
                
                process_questions = st.session_state.generator.generate_process_questions(
                    concepts['processes'], num_processes
                )
                
                analytical_questions = st.session_state.generator.generate_analytical_questions(
                    all_sentences, num_analytical
                )
                
                # Step 5: Validation
                status_text.info("✅ Validating question quality...")
                progress_bar.progress(80)
                
                all_questions = {
                    'fill_in_blank': st.session_state.generator.validate_questions(fill_blank_questions),
                    'definitions': st.session_state.generator.validate_questions(definition_questions),
                    'processes': st.session_state.generator.validate_questions(process_questions),
                    'analytical': st.session_state.generator.validate_questions(analytical_questions)
                }
                
                # Filter by difficulty
                for q_type in all_questions:
                    all_questions[q_type] = [q for q in all_questions[q_type] 
                                           if q.get('difficulty', 'Medium') in difficulty_filter]
                
                progress_bar.progress(100)
                status_text.success("🎉 Question generation completed!")
                
                # Store results
                st.session_state.all_questions = all_questions
                st.session_state.concepts = concepts
                st.session_state.segments = segments
    
    # Display Results
    if hasattr(st.session_state, 'all_questions'):
        st.header("📋 Generated Questions")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Fill-in-Blank", len(st.session_state.all_questions['fill_in_blank']))
        with col2:
            st.metric("Definitions", len(st.session_state.all_questions['definitions']))
        with col3:
            st.metric("Processes", len(st.session_state.all_questions['processes']))
        with col4:
            st.metric("Analytical", len(st.session_state.all_questions['analytical']))
        
        # Question display tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔤 Fill-in-Blank", "📖 Definitions", "⚙️ Processes", "🧮 Analytical", "📊 Analysis"
        ])
        
        with tab1:
            st.subheader("Fill-in-the-Blank Questions")
            for q in st.session_state.all_questions['fill_in_blank']:
                with st.expander(f"Question {q['id']} ({q.get('difficulty', 'Medium')})"):
                    st.write(f"**Question:** {q['question']}")
                    
                    for opt_key, opt_value in q['options'].items():
                        if opt_key == q['correct_answer']:
                            st.write(f"**{opt_key}) {opt_value}** ✅")
                        else:
                            st.write(f"{opt_key}) {opt_value}")
                    
                    if include_explanations and 'explanation' in q:
                        st.info(f"**Explanation:** {q['explanation']}")
                    
                    if include_source and 'source' in q:
                        st.text_area("Source Text", q['source'], height=100, key=f"source_{q['id']}")
        
        with tab2:
            st.subheader("Definition Questions")
            for q in st.session_state.all_questions['definitions']:
                with st.expander(f"Question {q['id']}"):
                    st.write(f"**Question:** {q['question']}")
                    if include_source:
                        st.text_area("Reference Text", q.get('expected_answer', ''), height=100, key=f"def_{q['id']}")
        
        with tab3:
            st.subheader("Process Questions")
            for q in st.session_state.all_questions['processes']:
                with st.expander(f"Question {q['id']}"):
                    st.write(f"**Question:** {q['question']}")
                    st.write(f"**Context:** {q.get('context', '')}")
                    if 'expected_elements' in q and q['expected_elements']:
                        st.write("**Key Elements to Include:**")
                        for element in q['expected_elements']:
                            st.write(f"• {element}")
        
        with tab4:
            st.subheader("Analytical Questions")
            for q in st.session_state.all_questions['analytical']:
                with st.expander(f"Question {q['id']} ({q.get('difficulty', 'Hard')})"):
                    st.write(f"**Question:** {q['question']}")
                    st.write(f"**Analysis Type:** {q.get('analysis_type', 'General')}")
                    if include_source:
                        st.text_area("Source Context", q.get('source', ''), height=100, key=f"anal_{q['id']}")
        
        with tab5:
            st.subheader("Content Analysis")
            
            # Concept analysis
            st.write("**Key Concepts Extracted:**")
            concepts = st.session_state.concepts
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Important Terms:**")
                st.write(", ".join(concepts['important_terms'][:20]))
                
                st.write("**Named Entities:**")
                st.write(", ".join(concepts['named_entities'][:10]))
            
            with col2:
                st.write("**Definitions Found:**")
                st.write(f"{len(concepts['definitions'])} definition statements")
                
                st.write("**Process Descriptions:**")
                st.write(f"{len(concepts['processes'])} process descriptions")
            
            # Document structure
            st.write("**Document Structure:**")
            for segment_name, sentences in st.session_state.segments.items():
                st.write(f"• {segment_name}: {len(sentences)} sentences")
        
        # Export functionality
        st.header("💾 Export Questions")
        
        if st.button("🔽 Generate Export File", type="primary"):
            # Prepare export data
            export_data = {
                'metadata': {
                    'generated_on': datetime.now().isoformat(),
                    'source_file': uploaded_file.name,
                    'total_questions': sum(len(questions) for questions in st.session_state.all_questions.values()),
                    'settings': {
                        'fill_blank_count': num_fill_blank,
                        'definition_count': num_definitions,
                        'process_count': num_processes,
                        'analytical_count': num_analytical,
                        'difficulty_filter': difficulty_filter
                    }
                },
                'questions': st.session_state.all_questions,
                'concepts': st.session_state.concepts
            }
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if export_format == "JSON":
                st.download_button(
                    label="📄 Download JSON",
                    data=json.dumps(export_data, indent=2, ensure_ascii=False),
                    file_name=f"ncert_questions_{timestamp}.json",
                    mime="application/json"
                )
            
            elif export_format == "Excel":
                # Create Excel file with multiple sheets
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Fill-in-blank questions
                    if st.session_state.all_questions['fill_in_blank']:
                        fb_df = pd.DataFrame([
                            {
                                'Question_ID': q['id'],
                                'Question': q['question'],
                                'Option_A': q['options']['A'],
                                'Option_B': q['options']['B'],
                                'Option_C': q['options']['C'],
                                'Option_D': q['options']['D'],
                                'Correct_Answer': q['correct_answer'],
                                'Difficulty': q.get('difficulty', 'Medium'),
                                'Source': q.get('source', '') if include_source else ''
                            }
                            for q in st.session_state.all_questions['fill_in_blank']
                        ])
                        fb_df.to_excel(writer, sheet_name='Fill_in_Blank', index=False)
                    
                    # Other question types
                    for q_type, questions in st.session_state.all_questions.items():
                        if q_type != 'fill_in_blank' and questions:
                            df = pd.DataFrame(questions)
                            df.to_excel(writer, sheet_name=q_type.title(), index=False)
                
                st.download_button(
                    label="📊 Download Excel",
                    data=output.getvalue(),
                    file_name=f"ncert_questions_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            elif export_format == "Text":
                # Create formatted text file
                text_content = "NCERT QUESTION GENERATOR - DETAILED REPORT\n"
                text_content += "=" * 60 + "\n\n"
                text_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                text_content += f"Source file: {uploaded_file.name}\n"
                text_content += f"Total questions: {sum(len(questions) for questions in st.session_state.all_questions.values())}\n\n"
                
                # Fill-in-blank questions
                if st.session_state.all_questions['fill_in_blank']:
                    text_content += "FILL-IN-THE-BLANK QUESTIONS\n"
                    text_content += "-" * 40 + "\n\n"
                    
                    for q in st.session_state.all_questions['fill_in_blank']:
                        text_content += f"Q{q['id']}. {q['question']}\n"
                        for opt_key, opt_value in q['options'].items():
                            marker = " ✓" if opt_key == q['correct_answer'] else ""
                            text_content += f"   {opt_key}) {opt_value}{marker}\n"
                        text_content += f"   Difficulty: {q.get('difficulty', 'Medium')}\n"
                        if include_explanations and 'explanation' in q:
                            text_content += f"   Explanation: {q['explanation']}\n"
                        if include_source and 'source' in q:
                            text_content += f"   Source: {q['source'][:100]}...\n"
                        text_content += "\n"
                
                # Definition questions
                if st.session_state.all_questions['definitions']:
                    text_content += "\nDEFINITION QUESTIONS\n"
                    text_content += "-" * 40 + "\n\n"
                    
                    for q in st.session_state.all_questions['definitions']:
                        text_content += f"Q{q['id']}. {q['question']}\n"
                        if 'key_terms' in q:
                            text_content += f"   Key terms: {', '.join(q['key_terms'])}\n"
                        if include_source and 'expected_answer' in q:
                            text_content += f"   Reference: {q['expected_answer'][:200]}...\n"
                        text_content += "\n"
                
                # Process questions
                if st.session_state.all_questions['processes']:
                    text_content += "\nPROCESS QUESTIONS\n"
                    text_content += "-" * 40 + "\n\n"
                    
                    for q in st.session_state.all_questions['processes']:
                        text_content += f"Q{q['id']}. {q['question']}\n"
                        if 'context' in q:
                            text_content += f"   Context: {q['context'][:150]}...\n"
                        if 'expected_elements' in q and q['expected_elements']:
                            text_content += f"   Key elements: {len(q['expected_elements'])} points identified\n"
                        text_content += "\n"
                
                # Analytical questions
                if st.session_state.all_questions['analytical']:
                    text_content += "\nANALYTICAL QUESTIONS\n"
                    text_content += "-" * 40 + "\n\n"
                    
                    for q in st.session_state.all_questions['analytical']:
                        text_content += f"Q{q['id']}. {q['question']}\n"
                        text_content += f"   Type: {q.get('analysis_type', 'General analysis')}\n"
                        text_content += f"   Difficulty: {q.get('difficulty', 'Hard')}\n"
                        if include_source and 'source' in q:
                            text_content += f"   Source: {q['source'][:100]}...\n"
                        text_content += "\n"
                
                # Summary statistics
                text_content += "\nCONTENT ANALYSIS SUMMARY\n"
                text_content += "-" * 40 + "\n"
                concepts = st.session_state.concepts
                text_content += f"Important terms identified: {len(concepts['important_terms'])}\n"
                text_content += f"Definition statements found: {len(concepts['definitions'])}\n"
                text_content += f"Process descriptions found: {len(concepts['processes'])}\n"
                text_content += f"Named entities identified: {len(concepts['named_entities'])}\n"
                text_content += f"Numerical facts found: {len(concepts['numerical_facts'])}\n"
                
                st.download_button(
                    label="📝 Download Text Report",
                    data=text_content,
                    file_name=f"ncert_questions_report_{timestamp}.txt",
                    mime="text/plain"
                )
            
            elif export_format == "CSV":
                # Create CSV for fill-in-blank questions (most structured)
                if st.session_state.all_questions['fill_in_blank']:
                    csv_data = []
                    for q in st.session_state.all_questions['fill_in_blank']:
                        csv_data.append({
                            'Question_ID': q['id'],
                            'Question_Type': 'Fill_in_Blank',
                            'Question': q['question'],
                            'Option_A': q['options']['A'],
                            'Option_B': q['options']['B'],
                            'Option_C': q['options']['C'],
                            'Option_D': q['options']['D'],
                            'Correct_Answer': q['correct_answer'],
                            'Difficulty': q.get('difficulty', 'Medium'),
                            'Explanation': q.get('explanation', '') if include_explanations else '',
                            'Source': q.get('source', '')[:200] if include_source else ''
                        })
                    
                    # Add other question types as open-ended
                    question_id_counter = len(csv_data) + 1
                    for q_type, questions in st.session_state.all_questions.items():
                        if q_type != 'fill_in_blank':
                            for q in questions:
                                csv_data.append({
                                    'Question_ID': question_id_counter,
                                    'Question_Type': q_type.title(),
                                    'Question': q['question'],
                                    'Option_A': '',
                                    'Option_B': '',
                                    'Option_C': '',
                                    'Option_D': '',
                                    'Correct_Answer': '',
                                    'Difficulty': q.get('difficulty', 'Medium'),
                                    'Explanation': '',
                                    'Source': q.get('source', '')[:200] if include_source else ''
                                })
                                question_id_counter += 1
                    
                    df = pd.DataFrame(csv_data)
                    csv_string = df.to_csv(index=False)
                    
                    st.download_button(
                        label="📋 Download CSV",
                        data=csv_string,
                        file_name=f"ncert_questions_{timestamp}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No structured questions available for CSV export")

    # Additional features section
    if hasattr(st.session_state, 'all_questions'):
        st.header("🔍 Advanced Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Question Quality Metrics")
            
            # Calculate quality metrics
            total_questions = sum(len(questions) for questions in st.session_state.all_questions.values())
            high_quality_count = 0
            
            for q_type, questions in st.session_state.all_questions.items():
                for q in questions:
                    if (len(q.get('question', '')) > 20 and
                        'source' in q and
                        len(q['source']) > 30):
                        high_quality_count += 1
            
            quality_percentage = (high_quality_count / total_questions * 100) if total_questions > 0 else 0
            
            st.metric("Quality Score", f"{quality_percentage:.1f}%", 
                     help="Percentage of questions meeting quality criteria")
            
            # Difficulty distribution
            difficulty_counts = {'Easy': 0, 'Medium': 0, 'Hard': 0}
            for q_type, questions in st.session_state.all_questions.items():
                for q in questions:
                    diff = q.get('difficulty', 'Medium')
                    if diff in difficulty_counts:
                        difficulty_counts[diff] += 1
            
            st.write("**Difficulty Distribution:**")
            for diff, count in difficulty_counts.items():
                percentage = (count / total_questions * 100) if total_questions > 0 else 0
                st.write(f"• {diff}: {count} ({percentage:.1f}%)")
        
        with col2:
            st.subheader("🎯 Content Coverage Analysis")
            
            # Analyze content coverage
            concepts = st.session_state.concepts
            segments = st.session_state.segments
            
            st.metric("Document Sections", len(segments))
            st.metric("Unique Concepts", len(set(concepts['important_terms'])))
            st.metric("Definition Coverage", f"{len(concepts['definitions'])}")
            
            # Most frequently tested concepts
            all_question_text = ""
            for q_type, questions in st.session_state.all_questions.items():
                for q in questions:
                    all_question_text += q.get('question', '') + " "
            
            concept_mentions = {}
            for term in concepts['important_terms'][:20]:  # Top 20 terms
                mentions = all_question_text.lower().count(term.lower())
                if mentions > 0:
                    concept_mentions[term] = mentions
            
            if concept_mentions:
                st.write("**Most Tested Concepts:**")
                sorted_concepts = sorted(concept_mentions.items(), key=lambda x: x[1], reverse=True)
                for concept, count in sorted_concepts[:5]:
                    st.write(f"• {concept}: {count} questions")
        
        # Question preview/test mode
        st.subheader("🧪 Question Preview & Test Mode")
        
        if st.button("🎲 Generate Quick Test"):
            # Create a sample test with mixed question types
            sample_questions = []
            
            # Add some fill-in-blank questions
            if st.session_state.all_questions['fill_in_blank']:
                sample_questions.extend(
                    random.sample(
                        st.session_state.all_questions['fill_in_blank'], 
                        min(3, len(st.session_state.all_questions['fill_in_blank']))
                    )
                )
            
            # Add some definition questions
            if st.session_state.all_questions['definitions']:
                sample_questions.extend(
                    random.sample(
                        st.session_state.all_questions['definitions'], 
                        min(2, len(st.session_state.all_questions['definitions']))
                    )
                )
            
            st.session_state.sample_test = sample_questions
        
        if hasattr(st.session_state, 'sample_test'):
            st.write("**Sample Test Questions:**")
            for i, q in enumerate(st.session_state.sample_test, 1):
                with st.container():
                    st.write(f"**{i}. {q['question']}**")
                    
                    if q['type'] == 'fill_in_blank':
                        # Create radio buttons for MCQ
                        options_list = [f"{k}) {v}" for k, v in q['options'].items()]
                        selected = st.radio(
                            f"Choose your answer for question {i}:",
                            options_list,
                            key=f"test_q_{i}",
                            index=None
                        )
                        
                        if selected:
                            selected_letter = selected.split(')')[0]
                            if selected_letter == q['correct_answer']:
                                st.success("✅ Correct!")
                            else:
                                st.error(f"❌ Incorrect. The correct answer is {q['correct_answer']}")
                    
                    st.divider()

    # Footer with tips
    st.markdown("---")
    st.markdown("""
    ### 💡 Tips for Better Question Generation:
    
    - **Upload high-quality PDFs**: Clear, well-formatted PDFs produce better results
    - **Use complete chapters**: Full chapters provide better context than partial content  
    - **Adjust filters**: Use difficulty and quantity filters to match your needs
    - **Review generated questions**: Always review and edit questions before use
    - **Export multiple formats**: Different formats serve different purposes (Excel for editing, JSON for integration)
    
    ### 🎯 Question Types Generated:
    
    - **Fill-in-the-Blank**: Test specific terminology and concepts
    - **Definition Questions**: Assess understanding of key terms  
    - **Process Questions**: Evaluate comprehension of procedures and mechanisms
    - **Analytical Questions**: Challenge critical thinking and reasoning skills
    
    ### 🔧 Advanced NLP Features:
    
    - Named Entity Recognition for important terms
    - Part-of-speech tagging for grammatical accuracy
    - Semantic analysis for context-aware questions
    - Content segmentation for comprehensive coverage
    - Quality validation to ensure question reliability
    """)

if __name__ == "__main__":
    main()