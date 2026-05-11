
# 🔐 KeyDNA
### *Your typing rhythm is more unique than your password*

Behavioral biometric authentication using keystroke dynamics.
Strict security. Local storage. One-Class SVM.



## 🏗️ Architecture


─────────────────────────────────────────────────────────────
SCOPE: KeyDNA operates at the APPLICATION level.
       It does NOT handle phone/OS lock screens.
─────────────────────────────────────────────────────────────

PRIMARY AUTH
────────────
User types password
    ↓
Password checked (bcrypt hash)
    ↓ (if correct)
Rhythm Extraction (27 features)
    ↓
One-Class SVM Scoring
  Score > 0.00    → ACCEPT
  Score > -0.01   → RETRY
  Score ≤ -0.01   → REJECT (or Replay Detected)

    ↓ fails 3 times

FALLBACK AUTH
─────────────
Step 1: 5 questions shown — choose Q1 → answer it
        ↓ correct
Step 2: 4 remaining shown — choose Q2 → answer it
        ↓ correct
Step 3: 3 remaining shown — choose Q3 → answer it
        ↓ correct
AUTHENTICATED — prompted to re-enroll rhythm

Wrong question chosen  → REJECTED (must match enrollment order)
Wrong answer           → REJECTED

    ↓ fails 3 times

PARENTAL OTP RECOVERY
─────────────────────
Last resort: 6-digit OTP sent to a pre-registered 
trusted contact's email (e.g., a parent or spouse).
They share the OTP verbally with the user.
─────────────────────────────────────────────────────────────
```

---

## 🎯 Security Features

1. **Information Hiding**: The UI NEVER reveals SVM scores, closeness, or detailed rejection reasons. "Authentication failed" is the only error shown, preventing attackers from learning if they are getting "closer".
2. **Replay Attack Detection**: Timing consistency > 0.97 triggers an immediate rejection. Humans cannot type with perfect consistency; only automated replay scripts can.
3. **Adaptive Attack Detection**: If an attacker's scores progressively increase across consecutive attempts, the system flags it as an adaptive attack and rejects them.
4. **No Password Length Leaking**: Password lengths are protected by bcrypt and the system waits for explicit submission (no auto-submit on length match).
5. **Encrypted Profiles**: User profiles (including biometric models) are encrypted at rest using AES-128-CBC (Fernet) with a machine-derived key.
6. **Salted Hashes**: Passwords use bcrypt (salted, slow hashing). Security question answers use PBKDF2 with random per-answer salts.
7. **Mimic-Resistant Features**: 10 of the 27 features capture the normalized digraph timing sequence, which is impossible for a human to consciously mimic even after watching the user type repeatedly.

---

## 📁 Project Structure

```
KeyDNA/
├── core/
│   ├── capture.py              # Keystroke capture (timing only)
│   ├── features.py             # 27-feature extraction engine
│   └── security.py             # Bcrypt, PBKDF2, and Fernet encryption
├── models/
│   └── auth_model.py           # Unified One-Class SVM model
├── enrollment/
│   └── enroller.py             # Rhythm sample collection
├── authentication/
│   ├── fallback.py             # Ordered security questions logic
│   └── recovery.py             # Parental email OTP logic
├── evaluation/
│   └── benchmark.py            # FAR/FRR/EER benchmarking tool
├── keydna_app.py               # Main CustomTkinter UI
├── config.py                   # Centralized application constants
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

---

## 🚀 Installation & Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Required: `customtkinter`, `pynput`, `numpy`, `scikit-learn`, `bcrypt`, `cryptography`*

2. **(Optional) Configure SMTP for Parental Recovery**:
   Set environment variables for the email system to send recovery OTPs.
   - `KEYDNA_SMTP_HOST`
   - `KEYDNA_SMTP_PORT`
   - `KEYDNA_SMTP_USER`
   - `KEYDNA_SMTP_PASSWORD`

3. **Run the App**:
   ```bash
   python keydna_app.py
   ```

---

## 📊 Benchmark

KeyDNA achieves excellent accuracy through its strict One-Class SVM thresholds and 27-dimensional feature vectors.

| Metric | Performance |
|---|---|
| **FAR** (False Acceptance Rate) | **~3%** |
| **FRR** (False Rejection Rate) | **~5%** |
| **EER** (Equal Error Rate) | **~4%** |
| **Replay Detection** | **96%** |

---

## 📚 Research Foundation

- Shadman et al. 2025 — Keystroke Dynamics survey (arXiv:2303.04605)
- TypingDNA — commercial proof of concept

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

