🔐 KeyDNA — Behavioral Biometric Authentication

Your typing rhythm is your key.

KeyDNA is an AI-powered desktop authentication system that identifies users through keystroke dynamics — the unique way each person types. No passwords alone. No fingerprints. Just your natural typing behavior.
Show Image
Show Image
Show Image
Show Image
Show Image

🚀 Download & Run
No Python needed — just download and double-click.
👉 Download KeyDNA.exe

Works on Windows 10/11. No installation required.


✨ Features

🧠 AI-Powered Authentication — KNN model learns your unique typing pattern
🎭 Mood-Aware Recognition — Adapts to 4 typing moods: Relaxed, Focused, Stressed, Tired
⌨️ Real Keystroke Capture — Captures actual key press/release timing via pynput
🔄 4-Step Enrollment — Password + Rhythm + Backup PIN + Security Questions
🛡️ Replay Attack Detection — Detects and blocks automated/copied inputs
🔑 Fallback Authentication — PIN + 3 security questions if rhythm fails
⏱️ Session Timeout — Auto signs out after 5 minutes of inactivity
🌙 Dark UI — Clean, modern dark interface built with CustomTkinter


🎯 How It Works
User types password
        │
        ▼
Keystroke timing captured
(key press, release, flight time)
        │
        ▼
Feature extraction
(dwell time, flight time, typing speed)
        │
        ▼
KNN Model compares with enrolled profile
        │
        ▼
   ACCEPT or REJECT
The system doesn't just check what you type — it checks how you type it.

🖥️ Screenshots
Login ScreenEnrollmentSuccessEnter password with rhythm4-step setup processAuthentication confirmed

📁 Project Structure
KeyDNA/
│
├── keydna_app.py          # Main application entry point
│
├── core/
│   ├── capture.py         # Real-time keystroke capture
│   └── features.py        # Feature extraction from keystrokes
│
├── authentication/
│   ├── authenticator.py   # Main authentication engine
│   └── fallback.py        # Fallback PIN + security questions
│
├── enrollment/
│   └── enroller.py        # User enrollment & profile creation
│
├── models/
│   ├── knn_auth.py        # KNN authentication model
│   └── mood_classifier.py # Mood detection from typing patterns
│
├── evaluation/
│   └── benchmark.py       # Model performance evaluation
│
└── data/
    └── user_profile.json  # Stored user typing profile

🛠️ Run From Source
Prerequisites

Python 3.11+
Windows OS (required for pynput keyboard capture)

Installation
bash# Clone the repository
git clone https://github.com/Abhilashyadav15/KeyDNA.git
cd KeyDNA

# Install dependencies
pip install -r requirements.txt

# Run the app
python keydna_app.py
Dependencies
numpy>=1.24.0
scikit-learn>=1.3.0
customtkinter>=5.2.0
pynput>=1.7.6

🔐 Enrollment Process

Step 1 — Password : Set a strong password (shown as strength indicator)
Step 2 — Rhythm : Type your password 10 times in each of 4 moods
Step 3 — Backup PIN : Set a 6-12 digit emergency PIN
Step 4 — Security Questions : Choose and answer 3 security questions

After enrollment, the KNN model trains on your real typing data.

🧪 Authentication Flow
Enter password
      │
      ├─ Wrong password? → Failed attempt (3 max)
      │
      └─ Correct password
              │
              ├─ Rhythm verified? → ✅ ACCESS GRANTED
              │
              └─ Rhythm failed? → Failed attempt
                        │
                        └─ 3 failures → Fallback (PIN + Questions)

🤖 ML Models
ModelPurposeKNN AuthenticatorCompares typing sample against enrolled profiles using distance-based similarityMood ClassifierDetects current typing mood (Relaxed / Focused / Stressed / Tired)Random ForestSupports mood classification with higher accuracy

🛡️ Security Features

🔒 Passwords stored as SHA-256 hashes — never in plain text
🚫 Replay attack detection — blocks autofill and copied inputs
🕵️ Key anonymization — key content never stored, only timing
⏱️ Session timeout — auto logout after 5 minutes
🔑 Multi-layer fallback — PIN + 3 security questions


📊 Research Areas
This project relates to active research in:

Zero-password authentication
Continuous behavioral authentication
Typing DNA / keystroke biometrics
Emotion-aware security systems

Similar commercial products: TypingDNA

📄 License
This project is licensed under the MIT License — see LICENSE for details.

👨‍💻 Author
Abhilash Yadav

GitHub: @Abhilashyadav15



⭐ If you find this project interesting, please give it a star on GitHub!
