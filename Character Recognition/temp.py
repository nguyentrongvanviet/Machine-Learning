import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
import pytesseract
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import json
import os

class OCRPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Character Recognition & Predictor")
        self.root.geometry("1200x800")
        
        # Data storage
        self.history_data = []
        self.model = None
        self.accuracy_history = []
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create main frames
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - OCR Operations
        left_frame = ttk.LabelFrame(main_frame, text="OCR Operations", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right panel - Analytics & Predictions
        right_frame = ttk.LabelFrame(main_frame, text="Analytics & Predictions", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_ocr_panel(left_frame)
        self.setup_analytics_panel(right_frame)
        
    def setup_ocr_panel(self, parent):
        # Image upload section
        upload_frame = ttk.Frame(parent)
        upload_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(upload_frame, text="Upload Image", 
                  command=self.upload_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(upload_frame, text="Capture from Camera", 
                  command=self.capture_from_camera).pack(side=tk.LEFT, padx=5)
        
        # Image display
        self.image_label = ttk.Label(parent, text="No image selected")
        self.image_label.pack(pady=10)
        
        # OCR controls
        ocr_controls = ttk.Frame(parent)
        ocr_controls.pack(fill=tk.X, pady=5)
        
        ttk.Button(ocr_controls, text="Extract Text", 
                  command=self.extract_text).pack(side=tk.LEFT, padx=5)
        ttk.Button(ocr_controls, text="Save Result", 
                  command=self.save_result).pack(side=tk.LEFT, padx=5)
        
        # Results display
        self.result_text = tk.Text(parent, height=8, width=50)
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Confidence meter
        confidence_frame = ttk.Frame(parent)
        confidence_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(confidence_frame, text="Confidence:").pack(side=tk.LEFT)
        self.confidence_var = tk.StringVar(value="0%")
        ttk.Label(confidence_frame, textvariable=self.confidence_var).pack(side=tk.LEFT)
        
        self.confidence_bar = ttk.Progressbar(confidence_frame, orient=tk.HORIZONTAL, length=200)
        self.confidence_bar.pack(side=tk.LEFT, padx=5)
        
    def setup_analytics_panel(self, parent):
        # Model controls
        model_frame = ttk.Frame(parent)
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(model_frame, text="Train Prediction Model", 
                  command=self.train_model).pack(side=tk.LEFT, padx=2)
        ttk.Button(model_frame, text="Predict Accuracy", 
                  command=self.predict_accuracy).pack(side=tk.LEFT, padx=2)
        ttk.Button(model_frame, text="Show Analytics", 
                  command=self.show_analytics).pack(side=tk.LEFT, padx=2)
        
        # Prediction input
        input_frame = ttk.LabelFrame(parent, text="Prediction Input", padding=5)
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="Image Complexity (1-10):").grid(row=0, column=0, sticky=tk.W)
        self.complexity_var = tk.StringVar(value="5")
        ttk.Entry(input_frame, textvariable=self.complexity_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(input_frame, text="Text Length:").grid(row=1, column=0, sticky=tk.W)
        self.text_length_var = tk.StringVar(value="50")
        ttk.Entry(input_frame, textvariable=self.text_length_var, width=10).grid(row=1, column=1, padx=5)
        
        # Prediction result
        self.prediction_var = tk.StringVar(value="No prediction yet")
        ttk.Label(parent, textvariable=self.prediction_var, 
                 font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Analytics display
        self.analytics_text = tk.Text(parent, height=15, width=50)
        self.analytics_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.process_image_file(file_path)
            
    def process_image_file(self, file_path):
        try:
            image = Image.open(file_path)
            # Resize for display
            display_image = image.copy()
            display_image.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(display_image)
            
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            self.current_image = image
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            
    def extract_text(self):
        if not hasattr(self, 'current_image'):
            messagebox.showwarning("Warning", "Please select an image first")
            return
            
        try:
            # Perform OCR
            custom_config = r'--oem 3 --psm 6'
            extracted_text = pytesseract.image_to_string(self.current_image, config=custom_config)
            
            # Get confidence data
            data = pytesseract.image_to_data(self.current_image, output_type=pytesseract.Output.DICT)
            confidences = [float(conf) for conf in data['conf'] if float(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Display results
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(1.0, extracted_text)
            
            self.confidence_var.set(f"{avg_confidence:.1f}%")
            self.confidence_bar['value'] = avg_confidence
            
            # Store for analytics
            self.store_ocr_result(extracted_text, avg_confidence, len(extracted_text))
            
        except Exception as e:
            messagebox.showerror("Error", f"OCR failed: {str(e)}")
            
    def store_ocr_result(self, text, confidence, text_length):
        """Store OCR result for analytics and model training"""
        result_data = {
            'timestamp': pd.Timestamp.now(),
            'text_length': text_length,
            'confidence': confidence,
            'complexity': self.estimate_complexity(text),
            'success_rate': 1 if confidence > 70 else 0  # Binary success indicator
        }
        
        self.history_data.append(result_data)
        
    def estimate_complexity(self, text):
        """Estimate image complexity based on text characteristics"""
        # Simple heuristic - you can improve this
        complexity = 1
        if len(text) > 100:
            complexity += 2
        if any(char.isdigit() for char in text):
            complexity += 1
        if any(not char.isalnum() for char in text):
            complexity += 1
        return min(complexity, 10)
    
    def train_model(self):
        """Train linear regression model on historical data"""
        if len(self.history_data) < 10:
            messagebox.showwarning("Warning", "Need at least 10 data points to train model")
            return
            
        try:
            # Convert to DataFrame
            df = pd.DataFrame(self.history_data)
            
            # Prepare features and target
            X = df[['text_length', 'complexity']]
            y = df['confidence']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            self.accuracy_history.append({
                'timestamp': pd.Timestamp.now(),
                'train_score': train_score,
                'test_score': test_score,
                'data_points': len(df)
            })
            
            # Update analytics
            self.update_analytics_display(train_score, test_score)
            
            messagebox.showinfo("Success", 
                               f"Model trained successfully!\n"
                               f"Train R²: {train_score:.3f}\n"
                               f"Test R²: {test_score:.3f}")
                               
        except Exception as e:
            messagebox.showerror("Error", f"Model training failed: {str(e)}")
            
    def predict_accuracy(self):
        """Predict OCR accuracy for given inputs"""
        if self.model is None:
            messagebox.showwarning("Warning", "Please train the model first")
            return
            
        try:
            complexity = float(self.complexity_var.get())
            text_length = float(self.text_length_var.get())
            
            prediction = self.model.predict([[text_length, complexity]])[0]
            confidence = max(0, min(100, prediction))  # Clamp between 0-100
            
            self.prediction_var.set(f"Predicted Accuracy: {confidence:.1f}%")
            
            # Add to analytics
            self.analytics_text.insert(tk.END, 
                f"\nPrediction: Length={text_length}, Complexity={complexity} -> {confidence:.1f}% accuracy\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            
    def update_analytics_display(self, train_score, test_score):
        """Update analytics text display"""
        df = pd.DataFrame(self.history_data)
        
        analytics_info = f"""
=== OCR Analytics Dashboard ===

Data Summary:
- Total OCR operations: {len(self.history_data)}
- Average confidence: {df['confidence'].mean():.1f}%
- Average text length: {df['text_length'].mean():.1f} characters

Model Performance:
- Training R² Score: {train_score:.3f}
- Testing R² Score: {test_score:.3f}
- Model: Linear Regression

Recent Predictions:
"""
        self.analytics_text.delete(1.0, tk.END)
        self.analytics_text.insert(1.0, analytics_info)
        
    def show_analytics(self):
        """Show detailed analytics with matplotlib"""
        if len(self.history_data) < 5:
            messagebox.showwarning("Warning", "Not enough data for analytics")
            return
            
        df = pd.DataFrame(self.history_data)
        
        # Create analytics window
        analytics_window = tk.Toplevel(self.root)
        analytics_window.title("Advanced Analytics")
        analytics_window.geometry("1000x800")
        
        # Create matplotlib figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Confidence over time
        ax1.plot(df['timestamp'], df['confidence'], 'b-', alpha=0.7)
        ax1.set_title('OCR Confidence Over Time')
        ax1.set_ylabel('Confidence (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Text length vs Confidence scatter
        ax2.scatter(df['text_length'], df['confidence'], alpha=0.6)
        if self.model is not None:
            # Add regression line
            x_range = np.linspace(df['text_length'].min(), df['text_length'].max(), 100)
            y_pred = self.model.predict(np.column_stack([x_range, np.full(100, df['complexity'].mean())]))
            ax2.plot(x_range, y_pred, 'r-', linewidth=2, label='Regression Line')
        ax2.set_xlabel('Text Length')
        ax2.set_ylabel('Confidence (%)')
        ax2.set_title('Text Length vs Confidence')
        ax2.legend()
        
        # Plot 3: Complexity distribution
        ax3.hist(df['complexity'], bins=10, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Complexity Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Image Complexity Distribution')
        
        # Plot 4: Model accuracy history
        if self.accuracy_history:
            acc_df = pd.DataFrame(self.accuracy_history)
            ax4.plot(acc_df['timestamp'], acc_df['train_score'], 'g-', label='Train Score')
            ax4.plot(acc_df['timestamp'], acc_df['test_score'], 'r-', label='Test Score')
            ax4.set_xlabel('Training Time')
            ax4.set_ylabel('R² Score')
            ax4.set_title('Model Performance Over Time')
            ax4.legend()
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, analytics_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def save_result(self):
        """Save OCR results and model data"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if file_path:
                save_data = {
                    'history_data': self.history_data,
                    'accuracy_history': self.accuracy_history,
                    'model_coefficients': self.model.coef_.tolist() if self.model else None,
                    'model_intercept': self.model.intercept_ if self.model else None
                }
                
                with open(file_path, 'w') as f:
                    json.dump(save_data, f, indent=2, default=str)
                    
                messagebox.showinfo("Success", "Data saved successfully!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data: {str(e)}")
            
    def capture_from_camera(self):
        """Capture image from webcam"""
        try:
            import cv2
            
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            
            if ret:
                # Convert to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Save temporarily and process
                temp_path = "temp_capture.jpg"
                cv2.imwrite(temp_path, frame)
                self.process_image_file(temp_path)
                
            cap.release()
            
        except ImportError:
            messagebox.showerror("Error", "OpenCV not installed. Install with: pip install opencv-python")
        except Exception as e:
            messagebox.showerror("Error", f"Camera capture failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRPredictorApp(root)
    root.mainloop()