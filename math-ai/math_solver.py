import cv2
import mediapipe as mp
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st
import time

# Streamlit page configuration
st.set_page_config(layout="wide", page_title="Math Gesture Solver")

# Page header
st.title("Math Gesture Solver")
st.write("Use hand gestures to write and solve math problems!")

# Create layout
col1, col2 = st.columns([3,2])

# Camera feed column
with col1:
    run = st.checkbox('Start Camera', value=True)
    FRAME_WINDOW = st.image([])
    clear_button = st.button('Clear Canvas')

# Output column
with col2:
    st.subheader("Math Solution")
    output_text_area = st.empty()
    st.write("""
    Instructions:
    1. ☝️ Index finger up - Draw
    2. 👍 Thumb up - Clear canvas
    3. ✋ All fingers up (except pinky) - Solve
    """)

# Initialize Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Initialize Gemini AI
genai.configure(api_key="Add your own api key")  # Replace with your API key
model = genai.GenerativeModel('gemini-1.5-flash')

def initialize_camera():
    """Initialize the webcam."""
    cap = cv2.VideoCapture(0)  # Try 0 first, if not working try 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

def get_landmark_coords(hand_landmarks, image_shape):
    """Convert hand landmarks to pixel coordinates."""
    landmarks = []
    for landmark in hand_landmarks.landmark:
        x = int(landmark.x * image_shape[1])
        y = int(landmark.y * image_shape[0])
        landmarks.append([x, y])
    return landmarks

def count_fingers(landmarks):
    """Count number of fingers up."""
    fingers = []
    
    # Thumb
    if landmarks[4][0] < landmarks[3][0]:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # Other fingers
    for tip in [8, 12, 16, 20]:
        if landmarks[tip][1] < landmarks[tip-2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
    return fingers

def get_hand_info(img):
    """Process image and detect hands."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        landmarks = get_landmark_coords(results.multi_hand_landmarks[0], img.shape)
        fingers = count_fingers(landmarks)
        
        # Draw hand landmarks for visualization
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(0,0,255), thickness=2)
            )
        
        return fingers, landmarks
    return None

def draw(info, prev_pos, canvas):
    """Draw on canvas based on hand gestures."""
    fingers, landmarks = info
    current_pos = None
    
    if fingers == [0, 1, 0, 0, 0]:  # Index finger up for drawing
        current_pos = (landmarks[8][0], landmarks[8][1])
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, prev_pos, current_pos, (255, 0, 255), 10)
    elif fingers == [1, 0, 0, 0, 0]:  # Thumb up for clearing
        canvas = np.zeros_like(canvas)
    
    return current_pos, canvas

def send_to_AI(model, canvas, fingers):
    """Send canvas to Gemini AI for math problem solving."""
    try:
        if fingers == [1, 1, 1, 1, 0]:  # All fingers except pinky up
            # Convert BGR to RGB for PIL
            canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(canvas_rgb)
            
            # Send to Gemini AI
            response = model.generate_content([
                "Solve this handwritten math problem. Only return the solution.",
                pil_image
            ])
            
            return response.text
        return None
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    cap = initialize_camera()
    prev_pos = None
    canvas = None
    last_ai_call = 0
    ai_cooldown = 2  # Seconds between AI calls
    
    while run:
        success, img = cap.read()
        if not success:
            st.error("Error accessing camera!")
            break
            
        img = cv2.flip(img, 1)
        
        if canvas is None:
            canvas = np.zeros_like(img)
        
        if clear_button:
            canvas = np.zeros_like(img)
        
        info = get_hand_info(img)
        if info:
            fingers, landmarks = info
            prev_pos, canvas = draw(info, prev_pos, canvas)
            
            # Check for solve gesture with cooldown
            current_time = time.time()
            if current_time - last_ai_call >= ai_cooldown:
                solution = send_to_AI(model, canvas, fingers)
                if solution:
                    output_text_area.markdown(f"**Solution:**\n{solution}")
                    last_ai_call = current_time
        else:
            prev_pos = None
        
        # Combine camera feed with drawing canvas
        image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
        FRAME_WINDOW.image(image_combined, channels="BGR")
        
        # Check for window close
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
