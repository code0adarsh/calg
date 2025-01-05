# Math Gesture Solver 📷✋➗

Math Gesture Solver is an interactive application that allows users to write mathematical equations using hand gestures captured through a webcam. The app uses Mediapipe for hand tracking, OpenCV for drawing, and Google Gemini AI for solving handwritten math problems.

---

## Features 🚀

- **Real-time Hand Tracking**: Tracks hand gestures using Mediapipe's powerful hand-detection capabilities.
- **Interactive Drawing Canvas**: Use your hand gestures to draw mathematical problems on a virtual canvas.
- **AI-Powered Problem Solving**: Gemini AI processes handwritten equations and provides instant solutions.
- **Gesture Controls**:  
  - ☝️ **Index finger up**: Draw on the canvas.  
  - 👍 **Thumb up**: Clear the canvas.  
  - ✋ **All fingers up (except pinky)**: Solve the equation.  

---

## Requirements 📋

Make sure you have the following dependencies installed:

- Python 3.8+
- OpenCV (`cv2`)
- Mediapipe (`mediapipe`)
- Numpy (`numpy`)
- Streamlit (`streamlit`)
- Google Generative AI (`google.generativeai`)
- PIL (Pillow)

---

## Installation 🛠️

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/math-gesture-solver.git
   cd math-gesture-solver
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
3. Configure your Google Gemini API Key: Replace the placeholder API key in the code with your valid API key:

  ```python
genai.configure(api_key="your_google_api_key")
```

How to Run ▶️
Launch the application:

  ```bash 
streamlit run app.py
```


Open the app in your browser (usually at http://localhost:8501).

## Interact with the application:

Check Start Camera to activate the webcam feed.
Use gestures to draw, clear, and solve math equations.
Instructions ✋
Gesture Commands:
| Gesture                         | Action               |
|---------------------------------|----------------------|
| ☝️ Index finger up             | Draw on the canvas   |
| 👍 Thumb up                    | Clear the canvas     |
| ✋ All fingers except pinky up | Solve the equation   |

### Steps to Solve a Math Problem:
1. Use the **Index finger up** gesture to draw your math equation on the canvas.
2. To clear the canvas, show the **Thumb up** gesture.
3. When your equation is ready, show the **All fingers except pinky up** gesture to get the solution.

---

### How It Works 💡
1. **Hand Tracking**: Mediapipe detects the user's hand and identifies specific landmarks to determine gestures.
2. **Canvas Drawing**: OpenCV draws lines on a virtual canvas based on the hand's movements.
3. **AI Integration**: Captured equations are sent to Google Gemini AI for processing and solution generation.
4. **Real-time Feedback**: Results are displayed instantly in the app's output column.

---

### Limitations ⚠️
1. Requires a well-lit environment for accurate hand tracking.
2. The AI model depends on the accuracy of the handwritten equation.
3. Cooldown of 2 seconds between consecutive solve gestures.

---

### Future Improvements 🛠️
1. Support for multi-hand interaction.
2. Enhanced AI model for complex math problems.
3. Additional gesture controls for advanced features.

---

### Acknowledgments 🙌
1. **Mediapipe** for real-time hand tracking.
2. **OpenCV** for image processing.
3. **Google Generative AI** for AI-based equation solving.


Feel free to reach out with questions, feedback, or contributions!




