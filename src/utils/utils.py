# Function to clean and tokenize text
def clean_text(text):
    # Remove non-alphabetic characters and split into words
    words = re.findall(r'\b\w+\b', str(text).lower())
    return words

# Function to clean text, remove stopwords, and return processed text
def preprocess_text(text):
    words = clean_text(text)  # Tokenize and remove non-alphabetic characters
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)  # Reconstruct sentence

# Function to get POS tags for better lemmatization
def get_wordnet_pos(word):
    """Map POS tag to first character used by WordNetLemmatizer"""
    from nltk.corpus import wordnet
    from nltk.tag import pos_tag
    
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)  # Default to noun if unknown

# Function to clean and tokenize text
def clean_text(text):
    # Remove non-alphabetic characters and split into words
    words = re.findall(r'\b\w+\b', str(text).lower())
    return words

def get_tweet_embedding(words, model):
    vectors = [model.wv[word] for word in words if word in model.wv]
    return sum(vectors) / len(vectors) if vectors else np.zeros(model.vector_size)

# Function to evaluate and display model results
def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    # Train the model
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{class_report}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f"{model_name.replace(' ', '_').lower()}_confusion_matrix.png")
    plt.close()
    
    return {
        'model': model,
        'accuracy': accuracy,
        'training_time': train_time
    }

def estimate_size_mb(df):
    # Estimate the size in memory
    size_bytes = df.memory_usage(deep=True).sum()
    size_mb = size_bytes / (1024 * 1024)
    return size_mb