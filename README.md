# 🚀 Advanced Fatigue Life Predictor

Predict fatigue life curves (E–N, γ–N) easily and accurately using a physics-informed neural network (PINN).

🌐 **[Try the Live Demo](https://fatigue-life-prediction-6zfzg2ae9wdtnan3cutbyi.streamlit.app/)**

---

## 📦 How to Run Locally

1. Download the following files:
    - `main.ipynb`
    - `FatiguePredictor.py`  
      (👉 Note: Files like `main0331.ipynb`, `main0427.ipynb` are previous versions.)

2. Open `main.ipynb` and install the required libraries.

3. Run all the cells.  
   (This will generate three files: `best_fatigue_pinn_model.pth`, `scaler_X.pkl`, `scaler_y.pkl`.)

4. Open `FatiguePredictor.py`.

5. Install the required libraries for the app.

6. In the terminal, run:
    ```bash
    streamlit run FatiguePredictor.py
    ```

7. 🎉 Enjoy predicting fatigue life curves!

---

## 🛠️ Tech Stack

- Python 3.10+
- Streamlit
- PyTorch
- scikit-learn
- matplotlib
- pandas, numpy

---

## 📜 Version Info

- `main.ipynb`: Latest version  
- `main0331.ipynb`, `main0427.ipynb`: Previous development versions

---

## ✨ Future Plans

- [ ] Add alloy composition input
- [ ] Deploy public API
- [ ] Expand fatigue dataset

---

## 🤝 Contributions

Feel free to open an issue or submit a pull request if you find a bug or want to improve the project!
