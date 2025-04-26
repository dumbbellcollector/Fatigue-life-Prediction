# 🚀 Advanced Fatigue Life Prediction with Physics-Informed Neural Networks (PINNs)

Predict tensile (ε–N) and shear (γ–N) fatigue life curves directly from simple material properties using a physics-guided machine learning model.

🌐 **[👉 Try the Live Web App](https://fatigue-life-prediction-6zfzg2ae9wdtnan3cutbyi.streamlit.app/)**

---

## 📖 Overview

This project develops an advanced fatigue life prediction system based on a Physics-Informed Neural Network (PINN).  
It combines simple tensile test properties with physical fatigue models to predict strain-life (E–N) and shear strain-life (γ–N) behavior across various metallic materials.

**Highlights:**
- Predict fatigue parameters: σ′f, b, ε′f, c (tensile) and τ′f, b₀, γ′f, c₀ (shear)
- Generate both tensile (ε–N) and shear (γ–N) curves
- Support different shear conversion methods (von Mises, Maximum Principal Stress)
- Integrated physics-informed loss based on the Coffin–Manson relation
- Model trained targeting 100% inclusion within the 1.5× scatter band

---

## 📦 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/fatigue-life-predictor.git
cd fatigue-life-predictor
```

### 2. Install Required Libraries
```bash
pip install -r requirements.txt
```

> Required libraries:
> - streamlit>=1.25
> - torch>=2.0
> - joblib
> - numpy
> - pandas
> - matplotlib
> - scikit-learn

### 3. Run the Streamlit App
```bash
streamlit run FatiguePredictor.py
```

> ⚡ Make sure the following model files are present in the working directory:
> - `best_fatigue_pinn_model.pth`
> - `scaler_X.pkl`
> - `scaler_y.pkl`

---

## 📈 Quick Demo

| Tensile Mode (ε–N) | Shear Mode (γ–N) |
| :----------------: | :--------------: |
| ![Tensile Curve Example](path/to/tensile_example.png) | ![Shear Curve Example](path/to/shear_example.png) |

---

## 🛠️ File Structure

```text
├── main.ipynb                  # Model training and evaluation notebook
├── FatiguePredictor.py         # Streamlit GUI app
├── best_fatigue_pinn_model.pth # Trained model weights
├── scaler_X.pkl                # Input feature scaler
├── scaler_y.pkl                # Output target scaler
├── requirements.txt            # Python dependencies
```

---

## 🔥 Model Highlights

- Physics-based regularization enforcing the Coffin–Manson relation
- Dual-task learning: tensile and shear parameter prediction
- Scatter band evaluation (1.5×, 2× inclusion rates)
- Streamlit GUI for real-time fatigue curve generation

---

## 🕓 Changelog

| Date | Update Summary |
|:----:|:--------------|
| 2025.03.31 | Started development with TensorFlow. Physics constraints applied only to b and c range. |
| 2025.04.11 | Migrated to PyTorch and initiated Streamlit-based GUI app development. |
| 2025.04.14 | Extended model to predict both tensile and shear fatigue life (previously only tensile). |
| 2025.04.27 | Improved prediction accuracy by grouping conversion based on TS; enhanced Streamlit app GUI. |

---

## 🌟 Current Progress

- ✅ Physics-informed PINN model training
- ✅ Streamlit GUI prediction system
- ✅ Tensile and shear fatigue curve generation
- 🚧 Future: Experimental data upload & validation
- 🚧 Future: Batch prediction for multiple materials

---

## ✨ Future Directions

- Incorporating alloy composition and heat treatment effects
- Improving shear fatigue prediction using advanced stress criteria
- Deploying a public API for fatigue life prediction services

---

## 🤝 Contributions

Pull requests, suggestions, and issues are welcome!  
Feel free to improve the model, GUI, or documentation.

---

## 📜 License

This project is licensed under the MIT License.

---

## 📢 Acknowledgement

This project is part of a broader research effort aiming to accelerate fatigue design by combining machine learning with physics-based knowledge.

---

## 📫 Contact

For questions, collaboration, or discussions, feel free to reach out:

**Email:** amtkpe123@gmail.com

---
