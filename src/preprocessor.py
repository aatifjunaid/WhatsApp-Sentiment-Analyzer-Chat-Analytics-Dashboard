"""Phase 2: Text Preprocessing"""

import re
import string
from pathlib import Path
import sys
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import emoji
import spacy

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SLANG_DICT, STOPWORDS_LANG

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextPreprocessor:
    """Handle text preprocessing and normalization"""
    
    def __init__(self, normalize_emojis=True, remove_stopwords=False):
        self.normalize_emojis = normalize_emojis
        self.remove_stopwords = remove_stopwords
        self.stop_words = set(stopwords.words(STOPWORDS_LANG)) if remove_stopwords else set()
        
        # Load spaCy model for advanced NLP (optional)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def expand_abbreviations(self, text):
        """Expand common slang and abbreviations"""
        words = text.split()
        expanded_words = []
        
        for word in words:
            # Check if word is in slang dictionary
            expanded = SLANG_DICT.get(word.lower(), word)
            expanded_words.append(expanded)
        
        return ' '.join(expanded_words)
    
    def normalize_emojis_to_text(self, text):
        """Convert emojis to descriptive text"""
        if not self.normalize_emojis:
            return text
        
        def emoji_to_text(emoji_char):
            emoji_data = emoji.demojize(emoji_char)
            # Remove colons and replace underscores with spaces
            return emoji_data.strip(':').replace('_', ' ')
        
        return emoji.replace_emoji(text, replace=emoji_to_text)
    
    def remove_urls(self, text):
        """Remove URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def remove_mentions(self, text):
        """Remove @mentions from text"""
        return re.sub(r'@\w+', '', text)
    
    def remove_html_tags(self, text):
        """Remove HTML tags"""
        return re.sub(r'<[^>]+>', '', text)
    
    def remove_extra_whitespace(self, text):
        """Remove extra whitespace"""
        return ' '.join(text.split())
    
    def lowercase(self, text):
        """Convert to lowercase"""
        return text.lower()
    
    def remove_punctuation(self, text, keep_exclamation_question=True):
        """Remove punctuation"""
        if keep_exclamation_question:
            # Keep ! and ? as they can indicate sentiment
            punct = string.punctuation.replace('!', '').replace('?', '')
            translator = str.maketrans('', '', punct)
        else:
            translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    def remove_numbers(self, text):
        """Remove numbers"""
        return re.sub(r'\d+', '', text)
    
    def tokenize(self, text):
        """Tokenize text"""
        tokens = word_tokenize(text)
        return tokens
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from tokens"""
        if not self.remove_stopwords:
            return tokens
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize(self, text):
        """Lemmatize using spaCy"""
        if self.nlp is None:
            return text
        
        doc = self.nlp(text)
        return ' '.join([token.lemma_ for token in doc])
    
    def preprocess(self, text, full_pipeline=True):
        """Apply full preprocessing pipeline"""
        # Basic cleaning
        text = self.normalize_emojis_to_text(text)
        text = self.remove_html_tags(text)
        text = self.remove_urls(text)
        text = self.remove_mentions(text)
        text = self.remove_extra_whitespace(text)
        text = self.lowercase(text)
        text = self.expand_abbreviations(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_extra_whitespace(text)
        
        if full_pipeline:
            # Advanced NLP processing
            if self.nlp is not None:
                text = self.lemmatize(text)
            
            tokens = self.tokenize(text)
            if self.remove_stopwords:
                tokens = self.remove_stopwords(tokens)
            text = ' '.join(tokens)
        
        return text
    
    def preprocess_batch(self, texts, full_pipeline=True):
        """Preprocess multiple texts"""
        return [self.preprocess(text, full_pipeline=full_pipeline) for text in texts]


class WhatsAppParser:
    """Parse WhatsApp chat exports"""
    
    @staticmethod
    def parse_whatsapp_export(file_path):
        """Parse WhatsApp .txt export file"""
        messages = []
        
        # WhatsApp export format: [DD/MM/YYYY, HH:MM:SS] Sender: Message
        pattern = r'\[(\d{1,2}/\d{1,2}/\d{4}), (\d{1,2}:\d{2}(?::\d{2})?(?:\s[AP]M)?)\] (.+?): (.*)'
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    match = re.match(pattern, line)
                    if match:
                        date, time, sender, message = match.groups()
                        messages.append({
                            'date': date,
                            'time': time,
                            'sender': sender,
                            'message': message
                        })
        except Exception as e:
            print(f"Error parsing WhatsApp export: {e}")
        
        return messages


if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor()
    
    test_text = "OMG! I love this 😍😍😍 Check out https://example.com @user123 #amazing"
    print("Original:", test_text)
    print("Preprocessed:", preprocessor.preprocess(test_text))
