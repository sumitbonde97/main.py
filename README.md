import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

class FakeNewsDetector:
    def __init__(self):
        """Initialize the Fake News Detector"""
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'NaiveBayes': MultinomialNB(),
            'SVM': SVC(kernel='linear', random_state=42, probability=True)
        }
        self.best_model = None
        self.best_model_name = None
        self.stemmer = PorterStemmer()
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
    
    def preprocess_text(self, text):
        """
        Preprocess text data for training/prediction
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        stop_words = set(stopwords.words('english'))
        tokens = [self.stemmer.stem(token) for token in tokens if token not in stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def load_sample_data(self):
        """
        Create sample data for demonstration
        In a real scenario, you would load data from a CSV file or database
        """
        # Sample fake news data
        sample_data = {
            'text': [
                "Scientists discover miracle cure that doctors don't want you to know about",
                "Breaking: Local politician announces new infrastructure funding for city projects",
                "You won't believe what this celebrity did - shocking revelation inside",
                "Study shows correlation between exercise and improved mental health outcomes",
                "URGENT: Government planning secret operation against citizens next week",
                "New research published in medical journal shows promising cancer treatment results",
                "Celebrity endorses weight loss pill that makes you lose 50 pounds overnight",
                "University researchers develop new renewable energy technology with 40% efficiency improvement",
                "EXCLUSIVE: Aliens spotted in downtown area by multiple witnesses last night",
                "Local school district reports improved graduation rates following new education initiatives",
                "This one weird trick will make you rich - financial advisors hate it",
                "Climate change report shows rising temperatures affecting agricultural productivity",
                "Doctors discover patients are being cured by this ancient remedy",
                "City council approves budget allocation for public transportation improvements",
                "BREAKING: Conspiracy uncovered - they don't want you to see this information",
                "Medical breakthrough: New vaccine shows 95% efficacy in clinical trials"
            ],
            'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Fake, 0 = Real
        }
        
        return pd.DataFrame(sample_data)
    
    def train_models(self, data):
        """
        Train multiple models and select the best performer
        """
        print("Preprocessing text data...")
        # Preprocess the text
        data['processed_text'] = data['text'].apply(self.preprocess_text)
        
        # Split data
        X = data['processed_text']
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Vectorize the text
        print("Vectorizing text...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        best_accuracy = 0
        model_results = {}
        
        print("\nTraining models...")
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train the model
            model.fit(X_train_vec, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_vec)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            model_results[name] = {
                'accuracy': accuracy,
                'model': model,
                'predictions': y_pred,
                'y_test': y_test
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            # Update best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = model
                self.best_model_name = name
        
        print(f"\nBest Model: {self.best_model_name} with accuracy: {best_accuracy:.4f}")
        
        # Print detailed results for best model
        best_results = model_results[self.best_model_name]
        print(f"\nDetailed Results for {self.best_model_name}:")
        print("Classification Report:")
        print(classification_report(best_results['y_test'], best_results['predictions'], 
                                  target_names=['Real', 'Fake']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(best_results['y_test'], best_results['predictions']))
        
        return model_results
    
    def predict(self, text):
        """
        Predict if a news article is fake or real
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Vectorize
        text_vec = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.best_model.predict(text_vec)[0]
        probability = self.best_model.predict_proba(text_vec)[0]
        
        result = {
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': {
                'real': probability[0],
                'fake': probability[1]
            }
        }
        
        return result
    
    def save_model(self, filepath='fake_news_model.pkl'):
        """
        Save the trained model and vectorizer
        """
        if self.best_model is None:
            raise ValueError("No model to save. Please train the model first.")
        
        model_data = {
            'model': self.best_model,
            'vectorizer': self.vectorizer,
            'model_name': self.best_model_name
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='fake_news_model.pkl'):
        """
        Load a previously trained model
        """
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.best_model_name = model_data['model_name']
        print(f"Model loaded: {self.best_model_name}")

def main():
    """
    Main function to demonstrate the fake news detector
    """
    print("=== AI-Based Fake News Detection System ===\n")
    
    # Initialize detector
    detector = FakeNewsDetector()
    
    # Load sample data (in real scenario, load from CSV)
    print("Loading sample data...")
    data = detector.load_sample_data()
    print(f"Loaded {len(data)} articles")
    
    # Train models
    results = detector.train_models(data)
    
    # Test with sample articles
    print("\n" + "="*50)
    print("TESTING WITH SAMPLE ARTICLES")
    print("="*50)
    
    test_articles = [
        "Scientists have discovered an amazing breakthrough that will change everything",
        "The city council approved the new budget proposal after extensive deliberation",
        "SHOCKING: This secret government operation will blow your mind",
        "Research shows that regular exercise improves cardiovascular health"
    ]
    
    for i, article in enumerate(test_articles, 1):
        print(f"\nTest Article {i}: {article}")
        result = detector.predict(article)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence - Real: {result['confidence']['real']:.3f}, Fake: {result['confidence']['fake']:.3f}")
    
    # Save the model
    print(f"\nSaving model...")
    detector.save_model()
    
    print("\n" + "="*50)
    print("INTERACTIVE TESTING")
    print("="*50)
    print("Enter news articles to test (or 'quit' to exit):")
    
    while True:
        user_input = input("\nEnter news article: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_input:
            try:
                result = detector.predict(user_input)
                print(f"\nPrediction: {result['prediction']}")
                print(f"Confidence - Real: {result['confidence']['real']:.3f}, Fake: {result['confidence']['fake']:.3f}")
                
                if result['prediction'] == 'FAKE':
                    print("⚠️  This article appears to be FAKE NEWS!")
                else:
                    print("✅ This article appears to be LEGITIMATE NEWS.")
            except Exception as e:
                print(f"Error: {e}")

# Additional utility functions
def load_custom_dataset(filepath):
    """
    Load custom dataset from CSV file
    Expected columns: 'text', 'label' (0 for real, 1 for fake)
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Loaded {len(data)} articles from {filepath}")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def evaluate_model_performance(detector, test_data):
    """
    Evaluate model performance on test dataset
    """
    if detector.best_model is None:
        print("Model not trained yet!")
        return
    
    predictions = []
    for text in test_data['text']:
        result = detector.predict(text)
        predictions.append(1 if result['prediction'] == 'FAKE' else 0)
    
    accuracy = accuracy_score(test_data['label'], predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_data['label'], predictions, target_names=['Real', 'Fake']))

if __name__ == "__main__":
    main()
