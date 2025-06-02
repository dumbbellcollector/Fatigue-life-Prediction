# 🚀 Advanced Fatigue Life Predictor

Predict tensile (ε–N) and shear (γ–N) fatigue life curves from simple material properties or alloy composition using a physics-guided machine learning model.

🌐 **[👉 Try the Live Web App](https://fatigue-life-prediction-6zfzg2ae9wdtnan3cutbyi.streamlit.app/)** *(Note: Link might point to an older version if not updated recently)*

---

## 📖 Overview

This project presents an advanced fatigue life prediction system. It leverages a Physics-Informed Neural Network (PINN) to estimate strain-life (ε–N) and shear strain-life (γ–N) behavior for metallic materials. Users can input either standard monotonic tensile properties or the alloy's chemical composition to obtain fatigue life predictions.

**Key Features:**
- **Dual Input Modes:**
    - **Monotonic Properties:** Input E, YS, TS, HB, and Poisson's ratio.
    - **Alloy Composition:** Input wt% of key alloying elements (e.g., C, Mn, Cr, Mo, P, S) to internally estimate monotonic properties.
- Predicts key fatigue parameters: σ′f, b, ε′f, c (tensile).
- Derives shear fatigue parameters (τ′f, b₀, γ′f, c₀) using established conversion methods.
- Generates and visualizes both tensile (ε–N) and shear (γ–N) fatigue curves.
- Employs a hybrid loss function: data-driven for some parameters, physics-guided (e.g., Hardness Method) for others, and incorporates the Coffin-Manson relation.
- User-friendly Streamlit web application for interactive predictions.

---

## 📦 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/dumbbellcollector/fatigue-life-predictor.git
cd fatigue-life-predictor

2. Install Required Libraries

Ensure you have Python 3.8+ installed. Then, install the dependencies:

pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Required libraries:

streamlit>=1.25

torch>=2.0

joblib

numpy

pandas

matplotlib

scikit-learn

3. Run the Streamlit App

Navigate to the directory containing the app script and run:

streamlit run FatiguePredictor0529.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

(Replace FatiguePredictor0529.py with the latest main application script name if different.)

⚡ Ensure the following files are present in the same directory as the Streamlit app script:

best_fatigue_pinn_model.pth (Trained PyTorch model weights)

scaler_X.pkl (Scaler for input features)

scaler_y.pkl (Scaler for target fatigue parameters)

composition_to_properties.py (Module for calculating properties from composition)

📈 Quick Demo

Input Modes:

Monotonic Properties Input	Alloy Composition Input

![alt text]([Link_to_Monotonic_Input_Image.png])
	
![alt text]([Link_to_Composition_Input_Image.png])

Example Output Curves:

Tensile Mode (ε–N)	Shear Mode (γ–N)

![alt text]([Link_to_Tensile_Example_Image.png])
	
![alt text]([Link_to_Shear_Example_Image.png])

(Please replace [Link_to_..._Image.png] with actual paths to your demo images in the repository.)

🛠️ File Structure
.
├── FatiguePredictor0529.py         # Main Streamlit GUI application script
├── composition_to_properties.py    # Module for property estimation from composition
├── main0526Nfplot.ipynb            # Jupyter Notebook for model training and evaluation
├── best_fatigue_pinn_model.pth     # Trained model weights
├── scaler_X.pkl                    # Input feature scaler
├── scaler_y.pkl                    # Output target scaler (contains scalers and target_cols list)
├── requirements.txt                # Python dependencies
├── images/                           # (Optional) Directory for demo images
└── README.md                       # This README file
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Text
IGNORE_WHEN_COPYING_END
🔥 Model & Application Highlights

Hybrid PINN Approach: Combines data-driven learning with physics-based regularization (e.g., Coffin-Manson relation, empirical hardness methods for specific parameters).

Flexible Input: Accepts either direct monotonic properties or alloy composition for broader usability.

Comprehensive Output: Provides key tensile and shear fatigue parameters along with full ε–N and γ–N curves.

Shear Parameter Conversion: Implements UTS-dependent criteria (von Mises, Max Principal, Interpolation) for robust shear fatigue estimation.

Accuracy Benchmarking: Performance evaluated against established empirical methods using metrics like 2x scatter band inclusion rate.

Interactive GUI: Streamlit application allows for easy input, real-time predictions, and visualization.

🕓 Changelog
Date	Update Summary
2024.05.29+	Implemented "Alloy Composition Input" mode, enabling fatigue prediction directly from chemical composition. Added composition_to_properties.py module. Enhanced UI for new input mode. (Ongoing refinement)
2024.05.14	Reverted to a primarily data-driven approach for some parameters after evaluating a more heavily physics-constrained model.
2024.05.08	Experimented with incorporating traditional fatigue parameter estimation methods (Hardness Method, Universal Slope Method) more directly into the loss function.
2024.04.27	Improved prediction accuracy by refining shear conversion logic based on Tensile Strength (TS); enhanced Streamlit app GUI.
2024.04.14	Extended model to predict both tensile and shear fatigue life.
2024.04.11	Migrated model from TensorFlow to PyTorch; initiated Streamlit-based GUI development.
2024.03.31	Initial development started with TensorFlow. Physics constraints primarily focused on 'b' and 'c' parameter ranges.

(Note: Dates are assumed to be YYYY.MM.DD for consistency. Please adjust if your convention is different.)

🌟 Current Progress & Accuracy

✅ PINN model successfully trained and validated for predicting tensile fatigue parameters from monotonic properties.

✅ Streamlit GUI operational for both "Monotonic Properties Input" and "Alloy Composition Input" modes.

✅ Generation of tensile (ε–N) and shear (γ–N) fatigue curves with component breakdown (elastic/plastic).

✅ Achieved a 2x scatter band inclusion rate of 66.7% for fatigue life (2Nf) prediction (monotonic property mode) across diverse steel grades, comparable to or exceeding conventional empirical methods for specific alloy families.

🚧 Ongoing:

Further refinement of the "Alloy Composition Input" mode, including validation of the composition-to-property estimation accuracy.

Improvement of ε'f (fatigue ductility coefficient) prediction accuracy.

🚧 Future:

Allow users to upload experimental S-N or ε-N data for comparison or model fine-tuning.

Implement batch prediction capabilities for analyzing multiple materials or compositions simultaneously.

✨ Future Directions

Advanced Compositional Effects: Incorporate more sophisticated models for predicting monotonic properties from alloy composition, potentially including interaction terms and effects of minor elements or heat treatment (if data becomes available).

Uncertainty Quantification: Provide an estimation of uncertainty or confidence intervals for the predicted fatigue life.

Expanded Material Database: Train the model on a wider range of metallic materials beyond steels.

Public API: Develop and deploy a public API for programmatic access to the fatigue life prediction service.

🤝 Contributions

Pull requests, suggestions, and issues are highly welcome! Feel free to contribute by:

Improving the accuracy or robustness of the prediction models.

Enhancing the user interface and experience.

Expanding the documentation or adding more examples.

Adding support for new materials or features.

📜 License

This project is licensed under the MIT License. See the LICENSE file for more details (if you add one).

📢 Acknowledgement

This project is part of ongoing research aimed at leveraging machine learning and physics-based knowledge to accelerate and improve the accuracy of fatigue design and material selection processes. We acknowledge the work of R. Basan (2024) for providing a valuable benchmark for conventional methods.

📫 Contact

For questions, collaboration, or discussions, please feel free to reach out:

YeoJoon Yoon

Email: [goat@sogang.ac.kr]

GitHub: dumbbellcollector
