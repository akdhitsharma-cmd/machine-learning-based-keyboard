import json
import sqlite3
import uuid
from collections import defaultdict
from flask import Flask, render_template_string, request, jsonify, session

app = Flask(__name__)
app.secret_key = 'keyboard_ml_secret_key_change_in_production'

# -------------------------------
# 1. Database Setup (User-specific word learning)
# -------------------------------
def init_db():
    conn = sqlite3.connect('word_learning.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_word_stats (
            user_id TEXT,
            word TEXT,
            count INTEGER,
            PRIMARY KEY (user_id, word)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def get_user_id():
    """Get or create a unique user ID stored in session."""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return session['user_id']

def update_user_word(user_id, word):
    """Increment the usage count for a word for this user."""
    conn = sqlite3.connect('word_learning.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO user_word_stats (user_id, word, count)
        VALUES (?, ?, 1)
        ON CONFLICT(user_id, word) DO UPDATE SET count = count + 1
    ''', (user_id, word))
    conn.commit()
    conn.close()

def get_user_word_counts(user_id):
    """Retrieve all user-specific word counts."""
    conn = sqlite3.connect('word_learning.db')
    c = conn.cursor()
    c.execute('SELECT word, count FROM user_word_stats WHERE user_id = ?', (user_id,))
    rows = c.fetchall()
    conn.close()
    return {word: count for word, count in rows}

# -------------------------------
# 2. Base Language Model (Static word frequencies)
# -------------------------------
# A comprehensive base word frequency list.
# These scores represent general English word probabilities.
# The model combines this static knowledge with user-specific learning.
BASE_WORD_FREQ = {
    # Most common English words
    "the": 1000, "be": 950, "to": 920, "of": 890, "and": 870, "a": 850,
    "in": 820, "that": 800, "have": 780, "i": 760, "it": 740, "for": 720,
    "not": 700, "on": 680, "with": 660, "he": 640, "as": 620, "you": 600,
    "do": 580, "at": 560, "this": 540, "but": 520, "his": 500, "by": 480,
    "from": 460, "they": 440, "we": 420, "say": 400, "her": 380, "she": 360,
    "or": 340, "an": 320, "will": 300, "my": 280, "one": 260, "all": 240,
    "would": 220, "there": 200, "their": 180, "what": 160, "so": 140,
    "up": 120, "out": 100, "if": 90, "about": 85, "who": 80, "get": 75,
    "which": 70, "go": 65, "me": 60, "when": 55, "make": 50, "can": 48,
    "like": 45, "time": 43, "no": 40, "just": 38, "him": 35, "know": 33,
    "take": 30, "people": 28, "into": 25, "year": 23, "your": 20, "good": 18,
    "some": 16, "could": 15, "them": 14, "see": 13, "other": 12, "than": 11,
    "then": 10, "now": 9, "look": 8, "only": 7, "come": 6, "its": 5,
    "over": 5, "think": 5, "also": 4, "back": 4, "after": 4, "use": 4,
    "two": 4, "how": 4, "our": 4, "work": 4, "first": 4, "well": 4,
    "way": 4, "even": 4, "new": 4, "want": 4, "because": 4, "any": 4,
    "these": 4, "give": 4, "day": 4, "most": 4, "us": 3,
    
    # Technical / demo-relevant words
    "keyboard": 25, "machine": 22, "learning": 22, "model": 20,
    "guess": 18, "word": 16, "input": 15, "remember": 14,
    "selection": 13, "flask": 12, "python": 12, "prediction": 11,
    "hello": 10, "world": 10, "data": 9, "science": 8, "algorithm": 7,
    "neural": 7, "network": 7, "train": 6, "predict": 6, "feedback": 5,
    "online": 5, "adaptive": 4, "frequency": 4, "probability": 3,
    
    # Additional common words for better coverage
    "please": 10, "thank": 9, "you're": 8, "welcome": 7, "sorry": 6,
    "yes": 8, "no": 8, "maybe": 5, "help": 7, "error": 5, "success": 4,
    "build": 6, "create": 6, "develop": 5, "test": 5, "deploy": 4,
    "run": 5, "code": 6, "function": 5, "class": 5, "object": 4,
    "variable": 4, "loop": 4, "condition": 3, "database": 5, "server": 5,
    "client": 4, "browser": 4, "javascript": 4, "html": 4, "css": 4,
    "style": 4, "button": 5, "click": 5, "type": 5, "text": 5,
    "user": 6, "interface": 5, "experience": 4, "design": 4, "responsive": 3
}

# Add some plurals and common variations
for word in list(BASE_WORD_FREQ.keys()):
    if word + "s" not in BASE_WORD_FREQ and len(word) > 2:
        BASE_WORD_FREQ[word + "s"] = BASE_WORD_FREQ[word] * 0.6
    if word + "ing" not in BASE_WORD_FREQ and len(word) > 2:
        BASE_WORD_FREQ[word + "ing"] = BASE_WORD_FREQ[word] * 0.5

# Ensure all lowercase
BASE_WORD_FREQ = {k.lower(): v for k, v in BASE_WORD_FREQ.items()}

def get_predictions(prefix, user_id, top_n=4):
    """
    Machine learning prediction model:
    Score(word) = BaseFrequency(word) + USER_WEIGHT * UserCount(word)
    This combines general language knowledge with personalized learning.
    """
    if not prefix:
        return []
    
    prefix = prefix.lower()
    USER_WEIGHT = 12.0  # How much user preference influences predictions
    
    # Get user-specific learned counts
    user_counts = get_user_word_counts(user_id)
    
    # Collect all candidate words (from base vocabulary + user's learned words)
    candidates = set(BASE_WORD_FREQ.keys())
    candidates.update(user_counts.keys())
    
    # Score and filter candidates that start with the prefix
    scored_words = []
    for word in candidates:
        if word.startswith(prefix):
            base_score = BASE_WORD_FREQ.get(word, 0)
            user_score = user_counts.get(word, 0) * USER_WEIGHT
            total_score = base_score + user_score
            if total_score > 0:
                scored_words.append((total_score, word))
    
    # Sort by score descending and return top N
    scored_words.sort(reverse=True, key=lambda x: x[0])
    return [word for score, word in scored_words[:top_n]]

# -------------------------------
# 3. Flask Routes
# -------------------------------
@app.route('/')
def index():
    """Main page with virtual keyboard and ML word prediction."""
    html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
        <title>ML Keyboard - Adaptive Word Prediction</title>
        <style>
            * {
                box-sizing: border-box;
                user-select: none;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                margin: 0;
                padding: 20px;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .container {
                max-width: 900px;
                width: 100%;
                background: rgba(255,255,255,0.95);
                border-radius: 30px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.2);
                padding: 25px;
                backdrop-filter: blur(5px);
            }
            h1 {
                text-align: center;
                color: #4a5568;
                margin-top: 0;
                font-size: 1.8rem;
            }
            .sub {
                text-align: center;
                color: #718096;
                margin-bottom: 25px;
                font-size: 0.9rem;
            }
            .input-area {
                background: #f7fafc;
                border-radius: 20px;
                padding: 20px;
                margin-bottom: 20px;
                border: 2px solid #e2e8f0;
            }
            #textInput {
                width: 100%;
                padding: 15px;
                font-size: 1.5rem;
                font-family: monospace;
                border: 2px solid #cbd5e0;
                border-radius: 15px;
                outline: none;
                transition: all 0.2s;
                background: white;
            }
            #textInput:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102,126,234,0.2);
            }
            .predictions {
                display: flex;
                gap: 12px;
                flex-wrap: wrap;
                margin-top: 15px;
                justify-content: center;
            }
            .pred-btn {
                background: linear-gradient(135deg, #667eea, #764ba2);
                border: none;
                color: white;
                padding: 10px 20px;
                border-radius: 40px;
                font-size: 1.1rem;
                font-weight: bold;
                cursor: pointer;
                transition: transform 0.1s, box-shadow 0.2s;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .pred-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            }
            .pred-btn:active {
                transform: translateY(1px);
            }
            .keyboard {
                background: #edf2f7;
                border-radius: 20px;
                padding: 15px;
                margin-top: 10px;
            }
            .key-row {
                display: flex;
                justify-content: center;
                gap: 8px;
                margin-bottom: 10px;
                flex-wrap: wrap;
            }
            .key {
                background: white;
                border: none;
                width: 60px;
                height: 60px;
                font-size: 1.3rem;
                font-weight: bold;
                border-radius: 12px;
                cursor: pointer;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: all 0.05s linear;
                color: #2d3748;
            }
            .key:active {
                transform: scale(0.96);
                background: #e2e8f0;
            }
            .special-key {
                background: #cbd5e0;
                width: auto;
                padding: 0 20px;
            }
            .space-key {
                width: 200px;
                background: #a0aec0;
            }
            .clear-key {
                background: #fc8181;
                color: white;
            }
            .backspace-key {
                background: #fbbf24;
            }
            @media (max-width: 700px) {
                .key { width: 45px; height: 45px; font-size: 1rem; }
                .space-key { width: 120px; }
                .pred-btn { font-size: 0.9rem; padding: 8px 16px; }
                #textInput { font-size: 1.2rem; }
            }
            .info {
                text-align: center;
                margin-top: 20px;
                font-size: 0.8rem;
                color: #718096;
            }
            .badge {
                background: #e9d8fd;
                color: #553c9a;
                border-radius: 20px;
                padding: 5px 12px;
                font-size: 0.7rem;
                display: inline-block;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔮 Adaptive ML Keyboard</h1>
            <div class="sub">🧠 Model learns from your selections | <span class="badge">Online Learning Active</span></div>
            
            <div class="input-area">
                <input type="text" id="textInput" placeholder="Start typing..." autofocus>
                <div class="predictions" id="predictions">
                    <button class="pred-btn" disabled>✨ Predictions appear here</button>
                </div>
            </div>
            
            <div class="keyboard" id="keyboard">
                <!-- Keyboard rows will be generated by JS -->
            </div>
            <div class="info">
                💡 Click on word predictions to select them → The ML model REMEMBERS your choice for next time!<br>
                🎯 Virtual keyboard works exactly like a physical one. Try typing "mac", then select "machine"!
            </div>
        </div>

        <script>
            // -------------------------------
            // DOM Elements
            // -------------------------------
            const textInput = document.getElementById('textInput');
            const predictionsDiv = document.getElementById('predictions');
            
            // -------------------------------
            // Helper: Get current word (last word being typed)
            // -------------------------------
            function getCurrentWord() {
                const text = textInput.value;
                const lastSpace = text.lastIndexOf(' ');
                const current = lastSpace === -1 ? text : text.substring(lastSpace + 1);
                return current;
            }
            
            // -------------------------------
            // Update predictions from backend ML model
            // -------------------------------
            async function updatePredictions() {
                const prefix = getCurrentWord();
                if (prefix.length === 0) {
                    predictionsDiv.innerHTML = '<button class="pred-btn" disabled>✨ Start typing for predictions</button>';
                    return;
                }
                
                try {
                    const response = await fetch(`/suggest?prefix=${encodeURIComponent(prefix)}`);
                    const suggestions = await response.json();
                    
                    if (suggestions.length === 0) {
                        predictionsDiv.innerHTML = '<button class="pred-btn" disabled>🔍 No predictions</button>';
                    } else {
                        predictionsDiv.innerHTML = suggestions.map(word => 
                            `<button class="pred-btn" data-word="${word}">${word}</button>`
                        ).join('');
                        
                        // Attach click handlers to prediction buttons
                        document.querySelectorAll('.pred-btn[data-word]').forEach(btn => {
                            btn.addEventListener('click', async (e) => {
                                const selectedWord = btn.getAttribute('data-word');
                                await selectPrediction(selectedWord);
                            });
                        });
                    }
                } catch (err) {
                    console.error('Prediction error:', err);
                    predictionsDiv.innerHTML = '<button class="pred-btn" disabled>⚠️ Error</button>';
                }
            }
            
            // -------------------------------
            // Select a prediction: replace current word, send to ML model for learning
            // -------------------------------
            async function selectPrediction(selectedWord) {
                const currentText = textInput.value;
                const currentWord = getCurrentWord();
                if (currentWord === '') return;
                
                // Replace the current word in the text
                const lastSpaceIdx = currentText.lastIndexOf(' ');
                let newText;
                if (lastSpaceIdx === -1) {
                    newText = selectedWord;
                } else {
                    newText = currentText.substring(0, lastSpaceIdx + 1) + selectedWord;
                }
                textInput.value = newText;
                
                // Tell the backend to remember this selection (update user-specific ML model)
                await fetch('/learn', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ word: selectedWord, prefix: currentWord })
                });
                
                // Refresh predictions after learning
                updatePredictions();
                // Keep focus on input
                textInput.focus();
            }
            
            // -------------------------------
            // Virtual Keyboard Construction
            // -------------------------------
            const layout = [
                ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
                ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
                ['z', 'x', 'c', 'v', 'b', 'n', 'm']
            ];
            
            function buildKeyboard() {
                const keyboardDiv = document.getElementById('keyboard');
                keyboardDiv.innerHTML = '';
                
                // Letter rows
                layout.forEach(row => {
                    const rowDiv = document.createElement('div');
                    rowDiv.className = 'key-row';
                    row.forEach(letter => {
                        const btn = document.createElement('button');
                        btn.textContent = letter;
                        btn.className = 'key';
                        btn.addEventListener('click', () => {
                            textInput.value += letter;
                            textInput.dispatchEvent(new Event('input'));
                            textInput.focus();
                        });
                        rowDiv.appendChild(btn);
                    });
                    keyboardDiv.appendChild(rowDiv);
                });
                
                // Special keys row
                const specialRow = document.createElement('div');
                specialRow.className = 'key-row';
                
                const spaceBtn = document.createElement('button');
                spaceBtn.textContent = '␣ SPACE';
                spaceBtn.className = 'key special-key space-key';
                spaceBtn.addEventListener('click', () => {
                    textInput.value += ' ';
                    textInput.dispatchEvent(new Event('input'));
                    textInput.focus();
                });
                
                const backspaceBtn = document.createElement('button');
                backspaceBtn.textContent = '⌫ BACK';
                backspaceBtn.className = 'key special-key backspace-key';
                backspaceBtn.addEventListener('click', () => {
                    textInput.value = textInput.value.slice(0, -1);
                    textInput.dispatchEvent(new Event('input'));
                    textInput.focus();
                });
                
                const clearBtn = document.createElement('button');
                clearBtn.textContent = '🗑 CLEAR';
                clearBtn.className = 'key special-key clear-key';
                clearBtn.addEventListener('click', () => {
                    textInput.value = '';
                    textInput.dispatchEvent(new Event('input'));
                    textInput.focus();
                });
                
                specialRow.appendChild(backspaceBtn);
                specialRow.appendChild(spaceBtn);
                specialRow.appendChild(clearBtn);
                keyboardDiv.appendChild(specialRow);
            }
            
            // -------------------------------
            // Event Listeners
            // -------------------------------
            textInput.addEventListener('input', () => {
                updatePredictions();
            });
            
            // Also allow physical keyboard input (to keep in sync)
            textInput.addEventListener('keyup', () => {
                updatePredictions();
            });
            
            // Initialize
            buildKeyboard();
            updatePredictions();
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)

@app.route('/suggest')
def suggest():
    """ML prediction endpoint: returns word suggestions for a given prefix."""
    prefix = request.args.get('prefix', '').strip()
    user_id = get_user_id()
    predictions = get_predictions(prefix, user_id)
    return jsonify(predictions)

@app.route('/learn', methods=['POST'])
def learn():
    """Endpoint to update the ML model with user's selection."""
    data = request.get_json()
    word = data.get('word', '').strip().lower()
    if not word:
        return jsonify({'status': 'error', 'message': 'No word provided'}), 400
    
    user_id = get_user_id()
    update_user_word(user_id, word)
    
    # Optional: Also learn if word is completely new, base model will adapt
    return jsonify({'status': 'success', 'message': f'Model remembered: {word}'})

# -------------------------------
# 4. Run the App
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)