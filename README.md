# Pneumonia Detection from Chest X-rays 🫁

A deep learning web application for detecting pneumonia from chest X-ray images using a Convolutional Neural Network (CNN).  
Developed as part of the **AI and Data Science Scholarship** from **Orange Digital Center** in collaboration with **Amit Learning**.

👉 **Live App**: [Streamlit App](https://pneumonia-dataset-project.streamlit.app/)  
📁 **GitHub Repository**: [Pneumonia Detection Project](https://github.com/mohamedhosam4/odc-_-pneumonia-dataset-project)

---

## 🧠 Project Overview

This project uses a deep learning model trained to classify chest X-ray images as either **Pneumonia** or **Normal**. The model is based on a CNN architecture and offers real-time predictions through a Streamlit web interface.

Users can upload X-ray images and get predictions on whether signs of pneumonia are detected.

---

## 🖼 Sample Screenshots

| Upload Interface | Prediction Result |
|------------------|-------------------|
| ![upload](https://github.com/mohamedhosam4/odc-_-pneumonia-dataset-project/blob/main/Screenshot%202025-05-06%20013933.png) | ![result](https://github.com/mohamedhosam4/odc-_-pneumonia-dataset-project/blob/main/Screenshot%202025-05-06%20014044.png) |

---

## 🔧 Features

- 📤 Upload a chest X-ray image.
- 📊 Get instant predictions (Pneumonia / Normal).
- 🧠 Powered by a CNN trained on labeled X-ray datasets.
- 💡 Easy-to-use Streamlit interface.

---

## 🚀 How to Run Locally

1. **Clone the repository**:

   ```bash
   git clone https://github.com/mohamedhosam4/odc-_-pneumonia-dataset-project.git
   cd odc-_-pneumonia-dataset-project
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the app**:

   ```bash
   streamlit run app.py
   ```

---

## 📁 File Structure

```
odc-_-pneumonia-dataset-project/
├── app.py                # Streamlit app
├── model/                # Trained model file
├── utils.py              # Image preprocessing
├── requirements.txt      # Python packages
└── README.md             # Project overview
```

---

## 📚 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Streamlit
- OpenCV

---

## 🤝 Contributors

- **Mohamed Hosam** – [@mohamedhosam4](https://github.com/mohamedhosam4)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🌟 Acknowledgments

This project was developed during the **AI and Data Science Training** from **Orange Digital Center** and **Amit Learning**.  
Thanks to the instructors and fellow learners for their continuous support and feedback.
