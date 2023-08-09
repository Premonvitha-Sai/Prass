import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the CSV data
data = pd.read_csv("data.csv")

# Drop the "Name" column
data = data.drop(columns=["Name"])

# Separate features (X) and target variable (y)
X = data.drop(columns=["Role"])
y = data["Role"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an SVC model
svc = SVC(kernel='linear', C=1, random_state=42)

# Train the model
svc.fit(X_train_scaled, y_train)


# ... [Rest of the Streamlit UI code]


# Streamlit app
st.title("Identify your Core Potential")
st.markdown('<span style="color:green">This is a self-assessment form that analyzes core potential abilities</span>', unsafe_allow_html=True)


# ...

# Creativity & Innovation question and slider
st.markdown("<h3>Creativity & Innovation:</h3>", unsafe_allow_html=True)
st.markdown("<p>On a scale of 1 to 5, how often do you actively seek out and implement unique and imaginative solutions to challenges?</p>", unsafe_allow_html=True)
creativity = st.slider("", 1, 5, key="slider_creativity")

# Adaptability & Flexibility question and slider
st.markdown("<h3>Adapability & Flexibility:</h3>", unsafe_allow_html=True)
st.markdown("<p>On a scale of 1 to 5, how well do you embrace and navigate changes and unexpected situations in your work?</p>", unsafe_allow_html=True)
adaptability = st.slider("", 1, 5, key="slider_adaptability")

# Problem Solving Skills question and slider
st.markdown("<h3>Problem Solving Skills:</h3>", unsafe_allow_html=True)
st.markdown("<p>On a scale of 1 to 5, how confident are you in your ability to analyze complex problems, break them down into manageable parts, and find effective solutions?</p>", unsafe_allow_html=True)
problem_solving = st.slider("", 1, 5, key="slider_problem_solving")

# Empathy & User Centric Focus question and slider
st.markdown("<h3>Empathy & User Centric Focus:</h3>", unsafe_allow_html=True)
st.markdown("<p>On a scale of 1 to 5, how consistently do you prioritize understanding and addressing the needs and emotions of users or stakeholders in your work?</p>", unsafe_allow_html=True)
empathy = st.slider("", 1, 5, key="slider_empathy")

# Iterative Mindset question and slider
st.markdown("<h3>Iterative Mindset:</h3>", unsafe_allow_html=True)
st.markdown("<p>On a scale of 1 to 5, how comfortable are you with receiving feedback, iterating on your ideas, and continuously refining your solutions?</p>", unsafe_allow_html=True)
iterative = st.slider("", 1, 5, key="slider_iterative")

# ... [add the rest of the sliders and questions in a similar way using unique keys]
# ...

# Leadership Influence question and slider
st.markdown("<h3>Leadership Influence:</h3>", unsafe_allow_html=True)
st.markdown("<p>On a scale of 1 to 5, how effectively do you inspire and guide others to achieve shared goals, and influence decisions within your team or organization?</p>", unsafe_allow_html=True)
leadership = st.slider("", 1, 5, key="slider_leadership")

# Critical Thinking question and slider
st.markdown("<h3>Critical Thinking:</h3>", unsafe_allow_html=True)
st.markdown("<p>On a scale of 1 to 5, how skilled are you at evaluating information, questioning assumptions, and making well-reasoned judgments?</p>", unsafe_allow_html=True)
critical_thinking = st.slider("", 1, 5, key="slider_critical_thinking")

# Continuous Learning question and slider
st.markdown("<h3>Continuous Learning:</h3>", unsafe_allow_html=True)
st.markdown("<p>On a scale of 1 to 5, how committed are you to regularly updating your knowledge and skills to stay current in your field?</p>", unsafe_allow_html=True)
continuous_learning = st.slider("", 1, 5, key="slider_continuous_learning")

# Resourcefulness question and slider
st.markdown("<h3>Resourcefulness:</h3>", unsafe_allow_html=True)
st.markdown("<p>On a scale of 1 to 5, how adept are you at finding creative and effective solutions within constraints and limitations?</p>", unsafe_allow_html=True)
resourcefulness = st.slider("", 1, 5, key="slider_resourcefulness")

# Submit button
submit_button = st.button("Submit")

# If the submit button is clicked, do the prediction
if submit_button:
    # Prepare input for prediction
    input_data = [[creativity, adaptability, problem_solving, empathy, iterative, leadership, critical_thinking, continuous_learning, resourcefulness]]
    scaled_input = scaler.transform(input_data)
    
    # Make a prediction using the trained model
    prediction = svc.predict(scaled_input)

    # Display prediction
    st.markdown(f"<h3>Your Predicted Role: <strong>{prediction[0]}</strong></h3>", unsafe_allow_html=True)
