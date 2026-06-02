# ⚡ QUICK START GUIDE - WhatsApp Sentiment Analysis

## 🚀 In 5 Minutes

### Step 1: Setup (2 minutes)

```bash
# Navigate to project folder
cd "path/to/New folder"

# Install dependencies
pip install -r requirements.txt

# Optional: Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Step 2: Get Data (1 minute)

**Option A - Kaggle API (Automatic):**
```bash
kaggle datasets download -d crowdflower/twitter-airline-sentiment
unzip twitter-airline-sentiment.zip -d data/
```

**Option B - Manual Download:**
1. Visit: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
2. Download `Tweets.csv`
3. Save to `data/` folder

### Step 3: Train Model (2 minutes)

```bash
python train_pipeline.py
```

**What happens:**
✓ Loads ~14,640 tweets  
✓ Preprocesses text (emojis, abbreviations, etc.)  
✓ Extracts TF-IDF features  
✓ Trains 2 models (Logistic Regression + SVM)  
✓ Evaluates and saves results  

**Output:**
- Trained models → `models/`
- Results & metrics → `outputs/`
- Confusion matrices → `outputs/`

---

## 📊 After Training - Make Predictions

### Single Prediction

```python
from src.inference import SentimentPredictor

predictor = SentimentPredictor('models/logistic_regression_final.pkl')

# Analyze a tweet
sentiment = predictor.predict_single("This product is amazing! 😍")
print(sentiment)  # Output: positive
```

### Batch Predictions

```python
texts = [
    "I love this!",
    "It's okay", 
    "Terrible experience"
]

results = predictor.predict_batch(texts, return_proba=True)

for r in results:
    print(f"'{r['text']}'")
    print(f"  Sentiment: {r['sentiment']}")
    print(f"  Confidence: {max(r['probabilities'].values()):.2%}")
```

### WhatsApp Chat Analysis

```python
# Export WhatsApp chat as .txt file
# Then:

df = predictor.predict_from_file('chat.txt', output_path='results.csv')
print(df.head())
```

**Output CSV has:**
- date, time, sender, message, sentiment, confidence

---

## 🔍 Explore Data

```python
from src.data_loader import DataLoader

loader = DataLoader()
loader.load_data()
loader.explore_data()  # Print statistics
loader.visualize_distribution(save_path='outputs/eda.png')
```

---

## 🧹 Text Processing Example

```python
from src.preprocessor import TextPreprocessor

preprocessor = TextPreprocessor()

# Original messy text
text = "OMG! I luv this 😍😍😍 Check https://example.com @user123 #amazing"

# Clean it
cleaned = preprocessor.preprocess(text, full_pipeline=True)
print(cleaned)
# Output: "oh my god i love this face with tears of joy"
```

---

## 📁 Project Structure

```
├── data/                      # Tweets.csv goes here
├── src/                       # Code modules
│   ├── data_loader.py        # Load & explore data
│   ├── preprocessor.py       # Clean text
│   ├── feature_extraction.py # TF-IDF vectors
│   ├── models.py             # Train ML models
│   ├── evaluation.py         # Evaluate results
│   └── inference.py          # Make predictions
├── models/                    # Trained models saved here
├── outputs/                   # Results & visualizations
├── config.py                  # Settings
├── train_pipeline.py          # Main training script
├── setup.py                   # Initialize project
├── examples.py                # Code examples
├── README.md                  # Full documentation
└── requirements.txt           # Dependencies
```

---

## 🎯 Expected Performance

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | ~75% | 0.75 |
| SVM | ~77% | 0.77 |

---

## ⚙️ Customize Configuration

Edit `config.py`:

```python
# Data split
TRAIN_SIZE = 0.7      # 70% training
VAL_SIZE = 0.15       # 15% validation
TEST_SIZE = 0.15      # 15% testing

# Preprocessing
NORMALIZE_EMOJIS = True       # Convert 😍 → "face with tears of joy"
LOWERCASE = True              # Convert to lowercase
REMOVE_URLS = True            # Remove links
REMOVE_MENTIONS = True        # Remove @mentions

# Abbreviations to expand
SLANG_DICT = {
    'u': 'you',
    'lol': 'laughing out loud',
    'omg': 'oh my god',
    # Add more...
}
```

---

## 🐛 Troubleshooting

**Error: ModuleNotFoundError: No module named 'kaggle'**
```bash
pip install kaggle
```

**Error: NLTK data missing**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

**Error: spaCy model not found**
```bash
python -m spacy download en_core_web_sm
```

**Large dataset taking too long?**
- Reduce `max_features` in `config.py` (default: 5000)
- Use smaller batch size
- Skip preprocessing pipeline step

---

## 📚 Examples

See `examples.py` for more code samples:
```bash
python examples.py
```

---

## 📖 Full Docs

See `README.md` for complete documentation

---

## ✅ Checklist

- [ ] Python 3.10+ installed
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] `data/Tweets.csv` exists
- [ ] Training completed: `python train_pipeline.py`
- [ ] Models saved in `models/` folder
- [ ] Can import and use predictor

---

## 🎓 Next Steps

1. **Understand the data:**
   ```python
   from src.data_loader import DataLoader
   DataLoader().load_data().explore_data()
   ```

2. **Visualize preprocessing:**
   ```python
   from src.preprocessor import TextPreprocessor
   pp = TextPreprocessor()
   print(pp.preprocess("Your messy text here"))
   ```

3. **Train models:**
   ```bash
   python train_pipeline.py
   ```

4. **Make predictions:**
   ```python
   from src.inference import SentimentPredictor
   p = SentimentPredictor('models/logistic_regression_final.pkl')
   print(p.predict_single("I love this!"))
   ```

5. **Analyze WhatsApp chats:**
   ```python
   df = p.predict_from_file('chat.txt', 'results.csv')
   ```

---

**Happy sentiment analyzing! 😊**

Questions? Check README.md or examples.py
