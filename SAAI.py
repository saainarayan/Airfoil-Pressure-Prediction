# ============================================================================
# NACA AIRFOIL CP PREDICTOR - STREAMLIT APPLICATION
# Phase 1: Configuration, Imports, and Constants
# ============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
from scipy.interpolate import interp1d
import io
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Try to import reportlab for PDF generation
try:
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.units import inch
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# ============================================================================
# APPLICATION CONFIGURATION CLASSES
# ============================================================================

@dataclass
class ModelConfig:
    """Neural network model configuration"""
    
    MODEL_FILENAME: str = 'best_airfoil_model.keras'
    INPUT_DIMENSIONS: int = 400  # 200 x,y coordinate pairs
    OUTPUT_DIMENSIONS: int = 200  # 100 upper + 100 lower Cp values
    
    @property
    def model_exists(self) -> bool:
        return os.path.exists(self.MODEL_FILENAME)

@dataclass
class AirfoilConfig:
    """NACA airfoil generation parameters"""
    
    # Generation parameters
    DEFAULT_NUM_POINTS: int = 200
    TARGET_PREDICTION_POINTS: int = 200
    CP_OUTPUT_POINTS: int = 200
    CP_UPPER_POINTS: int = 100
    CP_LOWER_POINTS: int = 100
    
    # Coordinate scaling bounds
    Y_GLOBAL_MIN: float = -0.15
    Y_GLOBAL_MAX: float = 0.15
    
    # NACA parameter limits
    NACA_CAMBER_MIN: int = 0
    NACA_CAMBER_MAX: int = 7
    NACA_POSITION_MIN: int = 2
    NACA_POSITION_MAX: int = 6
    NACA_THICKNESS_MIN: int = 6
    NACA_THICKNESS_MAX: int = 30
    
    # Validation
    VALID_NACA_DIGITS: int = 4


class VisualizationConfig:
    """Visualization and plotting settings"""
    
    # Figure settings
    FIGURE_DPI: int = 100
    FIGURE_SIZE_SINGLE: Tuple[int, int] = (14, 10)
    FIGURE_SIZE_COMPARISON: Tuple[int, int] = (18, 14)
    FIGURE_SIZE_PREVIEW: Tuple[int, int] = (14, 5)
    
    # Line and marker settings
    LINE_WIDTH_MAIN: int = 3
    LINE_WIDTH_DETAIL: int = 2
    MARKER_SIZE_MAIN: int = 4
    MARKER_SIZE_DETAIL: int = 3
    
    # Visual effects
    GRID_ALPHA: float = 0.3
    FILL_ALPHA: float = 0.3
    
    # Color schemes
    AIRFOIL_COLORS: List[str] = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA726', '#AB47BC', '#8D6E63']
    AIRFOIL_LINE_STYLES: List[str] = ['-', '--', '-.', ':', '-', '--']

@dataclass
class PerformanceConfig:
    """Performance analysis thresholds"""
    
    HIGH_SUCTION_THRESHOLD: float = 4.0
    MODERATE_SUCTION_THRESHOLD: float = 2.0
    MILD_SUCTION_THRESHOLD: float = 1.5
    HIGH_CP_RANGE_THRESHOLD: float = 4.0
    MODERATE_CP_RANGE_THRESHOLD: float = 2.0


class UIConfig:
    """User interface configuration"""
    
    PAGE_TITLE: str = "Neural Network Airfoil Cp Predictor"
    PAGE_ICON: str = "✈️"
    LAYOUT: str = "wide"
    SIDEBAR_STATE: str = "expanded"
    
    # Quiz configuration
    QUESTIONS_PER_LEVEL: int = 5
    PASSING_SCORE: int = 60
    EXCELLENT_SCORE: int = 80
    DIFFICULTY_LEVELS: List[str] = ["Beginner", "Intermediate", "Advanced"]

@st.cache_data
def load_and_resize_image(image_path, width, height):
    """Cache the image loading and resizing"""
    img = PILImage.open(image_path)
    return img.resize((width, height))
    
@dataclass
class ReportConfig:
    """PDF report generation settings"""
    
    # Page settings
    PAGE_SIZE = A4
    MARGIN: float = 72  # 1 inch margins
    
    # Font sizes
    TITLE_FONT_SIZE: int = 18
    HEADER_FONT_SIZE: int = 14
    BODY_FONT_SIZE: int = 11
    CAPTION_FONT_SIZE: int = 9
    
    # Colors
    HEADER_COLOR = colors.HexColor('#1f4e79')
    TABLE_HEADER_COLOR = colors.HexColor('#e6e6e6')
    
    # Report structure
    INCLUDE_TECHNICAL_DETAILS: bool = True
    INCLUDE_PERFORMANCE_ANALYSIS: bool = True
    INCLUDE_RECOMMENDATIONS: bool = True

# ============================================================================
# GLOBAL CONFIGURATION INSTANCES
# ============================================================================

# Create global configuration instances
MODEL_CFG = ModelConfig()
AIRFOIL_CFG = AirfoilConfig()
VIZ_CFG = VisualizationConfig()
PERF_CFG = PerformanceConfig()
UI_CFG = UIConfig()
REPORT_CFG = ReportConfig()

# ============================================================================
# ERROR MESSAGES AND VALIDATION
# ============================================================================

ERROR_MESSAGES = {
    'model_not_found': f"Model file '{MODEL_CFG.MODEL_FILENAME}' not found. Please ensure the file is in the current directory.",
    'model_load_failed': "Error loading model: {error}",
    'invalid_naca_code': "Invalid NACA code format. Must be 4 digits.",
    'coordinate_generation_failed': "Error generating airfoil coordinates: {error}",
    'prediction_failed': "Prediction failed: {error}",
    'export_failed': "Export operation failed: {error}",
    'pdf_not_available': "PDF generation requires 'pip install reportlab'"
}

# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=UI_CFG.PAGE_TITLE,
    page_icon=UI_CFG.PAGE_ICON,
    layout=UI_CFG.LAYOUT,
    initial_sidebar_state=UI_CFG.SIDEBAR_STATE
)

#from PIL import Image as PILImage

#resized_img = load_and_resize_image(r"C:\Users\Saai\Downloads\banner 9.png", 1200, 250)
#st.image(resized_img)


# ============================================================================
# PHASE 2: CORE DATA MODELS AND MODEL MANAGEMENT
# ============================================================================

class AirfoilData:
    """Core data class for storing airfoil information and analysis results"""
    
    def __init__(self, naca_code: str, camber: int, position: int, thickness: int):
        self.naca_code = naca_code
        self.camber = camber
        self.position = position
        self.thickness = thickness
        
        # Geometry data
        self.x_coords = None
        self.y_coords = None
        
        # Pressure coefficient data
        self.cp_distribution = None
        self.cp_upper = None
        self.cp_lower = None
        self.x_cp = None
        
        # Analysis metadata
        self.analysis_timestamp = datetime.now()
        self.prediction_successful = False
    
    @property
    def min_cp(self):
        """Get minimum pressure coefficient (peak suction)"""
        return self.cp_distribution.min() if self.cp_distribution is not None else None
    
    @property
    def max_cp(self):
        """Get maximum pressure coefficient"""
        return self.cp_distribution.max() if self.cp_distribution is not None else None
    
    @property
    def cp_range(self):
        """Get pressure coefficient range"""
        if self.cp_distribution is not None:
            return self.max_cp - self.min_cp
        return None
    
    @property
    def is_symmetric(self):
        """Check if airfoil is symmetric (zero camber)"""
        return self.camber == 0
    
    @property
    def thickness_ratio(self):
        """Get thickness ratio as decimal"""
        return self.thickness / 100.0
    
    @property
    def camber_ratio(self):
        """Get camber ratio as decimal"""
        return self.camber / 100.0
    
    @property
    def position_ratio(self):
        """Get camber position as decimal"""
        return self.position / 10.0 if self.camber > 0 else 0.0
    
    def get_summary_dict(self):
        """Get summary dictionary for export"""
        return {
            'NACA_Code': self.naca_code,
            'Camber_Percent': self.camber,
            'Position_Percent': self.position * 10 if self.camber > 0 else 0,
            'Thickness_Percent': self.thickness,
            'Min_Cp': self.min_cp,
            'Max_Cp': self.max_cp,
            'Cp_Range': self.cp_range,
            'Analysis_Time': self.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }

class ModelManager:
    """Manages neural network model loading and predictions"""
    
    def __init__(self):
        self.model = None
        self.is_loaded = False
        self.load_timestamp = None
        self.prediction_count = 0
    
    def load_model(self):
        """Load the trained neural network model"""
        try:
            if MODEL_CFG.model_exists:
                self.model = keras.models.load_model(MODEL_CFG.MODEL_FILENAME)
                self.is_loaded = True
                self.load_timestamp = datetime.now()
                return True, "Model loaded successfully!"
            else:
                return False, ERROR_MESSAGES['model_not_found']
        except Exception as e:
            error_msg = ERROR_MESSAGES['model_load_failed'].format(error=str(e))
            return False, error_msg
    
    def predict(self, scaled_input):
        """Make prediction using the loaded model"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        prediction = self.model.predict(scaled_input.reshape(1, -1), verbose=0)
        self.prediction_count += 1
        return prediction
    
    def get_model_info(self):
        """Get model information for display"""
        if not self.is_loaded:
            return "Model not loaded"
        
        return {
            'loaded': self.is_loaded,
            'load_time': self.load_timestamp.strftime('%Y-%m-%d %H:%M:%S') if self.load_timestamp else 'Unknown',
            'predictions_made': self.prediction_count,
            'input_shape': self.model.input_shape if self.model else None,
            'output_shape': self.model.output_shape if self.model else None
        }

@st.cache_resource
def get_model_manager():
    """Get cached model manager instance"""
    manager = ModelManager()
    success, message = manager.load_model()
    if not success:
    
    
        st.error(f"❌ {message}")
        st.info("💡 **Solution Steps:**\n"
                "1. Ensure 'best_airfoil_model.keras' is in the same directory as this script\n"
                "2. Check file permissions and integrity\n"
                "3. Verify TensorFlow installation")
    
    return manager, success

class NACAValidator:
    """Validates NACA parameters and codes"""
    
    @staticmethod
    def validate_naca_parameters(camber: int, position: int, thickness: int) -> tuple:
        """Validate NACA parameters and return validation result"""
        errors = []
        
        if not (AIRFOIL_CFG.NACA_CAMBER_MIN <= camber <= AIRFOIL_CFG.NACA_CAMBER_MAX):
            errors.append(f"Camber must be between {AIRFOIL_CFG.NACA_CAMBER_MIN} and {AIRFOIL_CFG.NACA_CAMBER_MAX}")
        
        if camber > 0 and not (AIRFOIL_CFG.NACA_POSITION_MIN <= position <= AIRFOIL_CFG.NACA_POSITION_MAX):
            errors.append(f"Position must be between {AIRFOIL_CFG.NACA_POSITION_MIN} and {AIRFOIL_CFG.NACA_POSITION_MAX}")
        
        if not (AIRFOIL_CFG.NACA_THICKNESS_MIN <= thickness <= AIRFOIL_CFG.NACA_THICKNESS_MAX):
            errors.append(f"Thickness must be between {AIRFOIL_CFG.NACA_THICKNESS_MIN} and {AIRFOIL_CFG.NACA_THICKNESS_MAX}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def generate_naca_code(camber: int, position: int, thickness: int) -> str:
        """Generate NACA code from parameters"""
        is_valid, errors = NACAValidator.validate_naca_parameters(camber, position, thickness)
        
        if not is_valid:
            raise ValueError(f"Invalid NACA parameters: {', '.join(errors)}")
        
        if camber == 0:
            return f"00{thickness:02d}"
        else:
            return f"{camber}{position}{thickness:02d}"
    
    @staticmethod
    def parse_naca_code(naca_code: str) -> tuple:
        """Parse NACA code into components"""
        if len(naca_code) != AIRFOIL_CFG.VALID_NACA_DIGITS:
            raise ValueError(ERROR_MESSAGES['invalid_naca_code'])
        
        try:
            camber = int(naca_code[0])
            position = int(naca_code[1]) if camber > 0 else 4
            thickness = int(naca_code[2:4])
            return camber, position, thickness
        except ValueError:
            raise ValueError(ERROR_MESSAGES['invalid_naca_code'])

class ErrorHandler:
    """Centralized error handling and user feedback"""
    
    @staticmethod
    def handle_model_loading_error(model_manager, model_loaded):
        """Handle model loading errors with user guidance"""
        if not model_loaded:
            st.error("🚨 Model Loading Failed")
            
            with st.expander("🔧 Troubleshooting Guide", expanded=True):
                st.markdown("""
                **Common Solutions:**
                
                1. **File Location**: Ensure `best_airfoil_model.keras` is in the same directory as this script
                
                2. **File Integrity**: The model file might be corrupted. Try re-downloading it
                
                3. **Permissions**: Check if the file has proper read permissions
                
                4. **TensorFlow Version**: Ensure you have a compatible TensorFlow version installed
                
                **File Requirements:**
                - Filename: exactly `best_airfoil_model.keras`
                - Size: Typically 10-100 MB for this type of model
                - Format: Keras SavedModel format
                """)
            
            return False
        return True
    
    @staticmethod
    def handle_coordinate_generation_error(error_msg: str):
        """Handle coordinate generation errors"""
        st.error(f"❌ Coordinate Generation Error: {error_msg}")
        st.info("💡 **Tip**: Check your NACA parameters and try again. Common issues include invalid parameter ranges.")
    
    @staticmethod
    def handle_prediction_error(error_msg: str):
        """Handle prediction errors"""
        st.error(f"❌ Prediction Error: {error_msg}")
        st.info("💡 **Tip**: Ensure the model is loaded correctly and input parameters are valid.")
    
    @staticmethod
    def display_success_message(message: str):
        """Display success message"""
        st.success(f"✅ {message}")
    
    @staticmethod
    def display_warning_message(message: str):
        """Display warning message"""
        st.warning(f"⚠️ {message}")
    
    @staticmethod
    def display_info_message(message: str):
        """Display info message"""
        st.info(f"ℹ️ {message}")

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

class SessionStateManager:
    """Manages Streamlit session state for the application"""
    
    DEFAULT_VALUES = {
        'page_mode': 'single',
        'analysis_complete': False,
        'last_analysis_time': None,
        'model_loaded': False,
        'quiz_session': None,
        'comparison_results': None
    }
    
    @staticmethod
    def initialize():
        """Initialize session state with default values"""
        for key, default_value in SessionStateManager.DEFAULT_VALUES.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def get(key: str, default=None):
        """Get session state value with optional default"""
        return st.session_state.get(key, default)
    
    @staticmethod
    def set(key: str, value):
        """Set session state value"""
        st.session_state[key] = value
    
    @staticmethod
    def clear_analysis_results():
        """Clear analysis-related session state"""
        keys_to_clear = ['analysis_complete', 'comparison_results', 'last_analysis_time']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

# ============================================================================
# PHASE 3: NACA AIRFOIL COORDINATE GENERATION AND PROCESSING
# ============================================================================

class CoordinateGenerator:
    """Handles NACA 4-digit airfoil coordinate generation"""
    
    @staticmethod
    @st.cache_data
    def generate_naca_coordinates(naca_code: str, num_points: int = None) -> tuple:
        """Generate NACA 4-digit airfoil coordinates using mathematical formulation"""
        if num_points is None:
            num_points = AIRFOIL_CFG.DEFAULT_NUM_POINTS
            
        try:
            # Validate and parse NACA code
            camber, position, thickness = NACAValidator.parse_naca_code(naca_code)
            
            # Convert to proper units for calculation
            m = camber / 100.0  # Maximum camber
            p = position / 10.0 if camber > 0 else 0.01  # Position of maximum camber
            t = thickness / 100.0  # Maximum thickness
            
            # Generate x coordinates using cosine distribution for better leading edge resolution
            beta = np.linspace(0, np.pi, num_points)
            x = (1 - np.cos(beta)) / 2
            
            # Calculate thickness distribution using NACA equation
            yt = CoordinateGenerator._calculate_thickness_distribution(x, t)
            
            # Calculate camber line and slope
            yc, dyc_dx = CoordinateGenerator._calculate_camber_line(x, m, p)
            
            # Calculate surface coordinates
            theta = np.arctan(dyc_dx)
            
            # Upper surface coordinates
            x_upper = x - yt * np.sin(theta)
            y_upper = yc + yt * np.cos(theta)
            
            # Lower surface coordinates
            x_lower = x + yt * np.sin(theta)
            y_lower = yc - yt * np.cos(theta)
            
            # Combine coordinates (upper surface from TE to LE, then lower surface from LE to TE)
            x_coords = np.concatenate([x_upper, x_lower[::-1]])
            y_coords = np.concatenate([y_upper, y_lower[::-1]])
            
            return x_coords, y_coords
            
        except Exception as e:
            error_msg = ERROR_MESSAGES['coordinate_generation_failed'].format(error=str(e))
            raise RuntimeError(error_msg)
    
    @staticmethod
    def _calculate_thickness_distribution(x: np.ndarray, t: float) -> np.ndarray:
        """Calculate NACA thickness distribution using standard coefficients"""
        # NACA thickness distribution coefficients
        a0, a1, a2, a3, a4 = 0.2969, -0.1260, -0.3516, 0.2843, -0.1015
        
        return 5 * t * (a0 * np.sqrt(x) + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4)
    
    @staticmethod
    def _calculate_camber_line(x: np.ndarray, m: float, p: float) -> tuple:
        """Calculate camber line and its derivative"""
        if m == 0:  # Symmetric airfoil
            yc = np.zeros_like(x)
            dyc_dx = np.zeros_like(x)
        else:
            yc = np.zeros_like(x)
            dyc_dx = np.zeros_like(x)
            
            # Forward section (0 <= x <= p)
            forward_mask = x <= p
            yc[forward_mask] = (m / p**2) * (2 * p * x[forward_mask] - x[forward_mask]**2)
            dyc_dx[forward_mask] = (2 * m / p**2) * (p - x[forward_mask])
            
            # Aft section (p < x <= 1)
            aft_mask = x > p
            yc[aft_mask] = (m / (1 - p)**2) * ((1 - 2*p) + 2*p*x[aft_mask] - x[aft_mask]**2)
            dyc_dx[aft_mask] = (2 * m / (1 - p)**2) * (p - x[aft_mask])
        
        return yc, dyc_dx

class CoordinateProcessor:
    """Handles coordinate processing for machine learning input"""
    
    @staticmethod
    @st.cache_data
    def scale_coordinates_for_prediction(x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        """Scale and process coordinates for neural network input"""
        target_points = AIRFOIL_CFG.TARGET_PREDICTION_POINTS
        
        # Calculate arc-length parameterization for consistent point distribution
        distances = CoordinateProcessor._calculate_arc_length(x_coords, y_coords)
        
        # Resample to target number of points
        x_resampled, y_resampled = CoordinateProcessor._resample_coordinates(
            x_coords, y_coords, distances, target_points
        )
        
        # Scale coordinates to [0,1] range for neural network
        x_scaled = CoordinateProcessor._scale_x_coordinates(x_resampled)
        y_scaled = CoordinateProcessor._scale_y_coordinates(y_resampled)
        
        # Interleave coordinates for ML input format
        scaled_coords = np.zeros(target_points * 2)
        scaled_coords[0::2] = x_scaled  # x coordinates at even indices
        scaled_coords[1::2] = y_scaled  # y coordinates at odd indices
        
        return scaled_coords
    
    @staticmethod
    def _calculate_arc_length(x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        """Calculate cumulative arc length along the airfoil perimeter"""
        distances = np.zeros(len(x_coords))
        for i in range(1, len(x_coords)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            distances[i] = distances[i-1] + np.sqrt(dx*dx + dy*dy)
        return distances
    
    @staticmethod
    def _resample_coordinates(x_coords: np.ndarray, y_coords: np.ndarray, 
                            distances: np.ndarray, target_points: int) -> tuple:
        """Resample coordinates to target number of points using interpolation"""
        total_length = distances[-1]
        target_distances = np.linspace(0, total_length, target_points)
        
        # Interpolate coordinates at target distances
        x_interp = interp1d(distances, x_coords, kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
        y_interp = interp1d(distances, y_coords, kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
        
        x_resampled = x_interp(target_distances)
        y_resampled = y_interp(target_distances)
        
        return x_resampled, y_resampled
    
    @staticmethod
    def _scale_x_coordinates(x_coords: np.ndarray) -> np.ndarray:
        """Scale x coordinates to [0,1] range"""
        x_min, x_max = x_coords.min(), x_coords.max()
        if x_max > x_min:
            return (x_coords - x_min) / (x_max - x_min)
        else:
            return x_coords
    
    @staticmethod
    def _scale_y_coordinates(y_coords: np.ndarray) -> np.ndarray:
        """Scale y coordinates to [0,1] range using global bounds"""
        return np.clip(
            (y_coords - AIRFOIL_CFG.Y_GLOBAL_MIN) / 
            (AIRFOIL_CFG.Y_GLOBAL_MAX - AIRFOIL_CFG.Y_GLOBAL_MIN), 
            0, 1
        )

class GeometryAnalyzer:
    """Analyzes airfoil geometry characteristics"""
    
    @staticmethod
    def calculate_geometric_properties(x_coords: np.ndarray, y_coords: np.ndarray) -> dict:
        """Calculate key geometric properties of the airfoil"""
        
        # Find leading and trailing edges
        le_idx = np.argmin(x_coords)
        te_idx = np.argmax(x_coords)
        
        # Calculate maximum thickness
        upper_y = y_coords[:len(y_coords)//2]
        lower_y = y_coords[len(y_coords)//2:]
        thickness_distribution = upper_y - lower_y[::-1]
        max_thickness = np.max(thickness_distribution)
        max_thickness_location = x_coords[np.argmax(thickness_distribution)]
        
        # Calculate area and perimeter
        area = GeometryAnalyzer._calculate_area(x_coords, y_coords)
        perimeter = GeometryAnalyzer._calculate_perimeter(x_coords, y_coords)
        
        # Leading edge radius approximation
        le_radius = GeometryAnalyzer._estimate_leading_edge_radius(x_coords, y_coords, le_idx)
        
        return {
            'max_thickness': max_thickness,
            'max_thickness_location': max_thickness_location,
            'area': area,
            'perimeter': perimeter,
            'leading_edge_radius': le_radius,
            'aspect_ratio': 1.0 / max_thickness,  # Assuming unit chord
            'fineness_ratio': 1.0 / max_thickness
        }
    
    @staticmethod
    def _calculate_area(x_coords: np.ndarray, y_coords: np.ndarray) -> float:
        """Calculate airfoil cross-sectional area using shoelace formula"""
        n = len(x_coords)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += x_coords[i] * y_coords[j]
            area -= x_coords[j] * y_coords[i]
        return abs(area) / 2.0
    
    @staticmethod
    def _calculate_perimeter(x_coords: np.ndarray, y_coords: np.ndarray) -> float:
        """Calculate airfoil perimeter"""
        perimeter = 0.0
        for i in range(len(x_coords) - 1):
            dx = x_coords[i+1] - x_coords[i]
            dy = y_coords[i+1] - y_coords[i]
            perimeter += np.sqrt(dx*dx + dy*dy)
        return perimeter
    
    @staticmethod
    def _estimate_leading_edge_radius(x_coords: np.ndarray, y_coords: np.ndarray, le_idx: int) -> float:
        """Estimate leading edge radius using curvature calculation"""
        # Use points around leading edge for radius estimation
        n_points = min(10, len(x_coords) // 10)
        start_idx = max(0, le_idx - n_points)
        end_idx = min(len(x_coords), le_idx + n_points)
        
        # Extract local coordinates
        local_x = x_coords[start_idx:end_idx]
        local_y = y_coords[start_idx:end_idx]
        
        # Fit circle to leading edge points and estimate radius
        if len(local_x) >= 3:
            try:
                # Simple radius estimation using three points
                x1, y1 = local_x[0], local_y[0]
                x2, y2 = local_x[len(local_x)//2], local_y[len(local_y)//2]
                x3, y3 = local_x[-1], local_y[-1]
                
                # Calculate radius using circumcircle formula
                d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
                if abs(d) > 1e-10:
                    ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
                    uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d
                    radius = np.sqrt((x1 - ux)**2 + (y1 - uy)**2)
                    return radius
            except:
                pass
        
        # Fallback: approximate using local curvature
        return 0.01  # Default small radius

class CoordinateValidator:
    """Validates airfoil coordinate quality and consistency"""
    
    @staticmethod
    def validate_coordinates(x_coords: np.ndarray, y_coords: np.ndarray) -> tuple:
        """Validate airfoil coordinates for quality and consistency"""
        errors = []
        warnings = []
        
        # Check array lengths
        if len(x_coords) != len(y_coords):
            errors.append("X and Y coordinate arrays must have equal length")
        
        # Check minimum number of points
        if len(x_coords) < AIRFOIL_CFG.DEFAULT_NUM_POINTS // 2:
            warnings.append(f"Coordinate array has fewer than {AIRFOIL_CFG.DEFAULT_NUM_POINTS // 2} points")
        
        # Check coordinate ranges
        if np.any(x_coords < -0.1) or np.any(x_coords > 1.1):
            warnings.append("X coordinates outside expected range [0, 1]")
        
        if np.any(y_coords < -0.5) or np.any(y_coords > 0.5):
            warnings.append("Y coordinates outside typical airfoil range [-0.5, 0.5]")
        
        # Check for NaN or infinite values
        if np.any(~np.isfinite(x_coords)) or np.any(~np.isfinite(y_coords)):
            errors.append("Coordinates contain NaN or infinite values")
        
        # Check for proper closure (first and last points should be close)
        if len(x_coords) > 1:
            closure_distance = np.sqrt((x_coords[0] - x_coords[-1])**2 + (y_coords[0] - y_coords[-1])**2)
            if closure_distance > 0.01:
                warnings.append("Airfoil may not be properly closed")
        
        # Check for self-intersection (basic check)
        if CoordinateValidator._has_self_intersection(x_coords, y_coords):
            errors.append("Airfoil coordinates have self-intersection")
        
        is_valid = len(errors) == 0
        return is_valid, errors, warnings
    
    @staticmethod
    def _has_self_intersection(x_coords: np.ndarray, y_coords: np.ndarray) -> bool:
        """Basic check for self-intersection in airfoil coordinates"""
        # Simple implementation - check if any line segments cross
        n = len(x_coords)
        if n < 4:
            return False
        
        # Check a subset of segments to avoid O(n²) complexity
        step = max(1, n // 50)
        for i in range(0, n - 1, step):
            for j in range(i + 2, min(n - 1, i + 20), step):
                if CoordinateValidator._segments_intersect(
                    x_coords[i], y_coords[i], x_coords[i+1], y_coords[i+1],
                    x_coords[j], y_coords[j], x_coords[j+1], y_coords[j+1]
                ):
                    return True
        return False
    
    @staticmethod
    def _segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4) -> bool:
        """Check if two line segments intersect"""
        def ccw(Ax, Ay, Bx, By, Cx, Cy):
            return (Cy - Ay) * (Bx - Ax) > (By - Ay) * (Cx - Ax)
        
        return (ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and
                ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4))

# ============================================================================
# PHASE 4: AIRCRAFT PRESETS AND CONFIGURATION MANAGEMENT
# ============================================================================

class AircraftPresets:
    """Manages aircraft preset configurations for different applications"""
    
    @staticmethod
    def get_presets():
        """Return comprehensive aircraft preset configurations"""
        return {
            "Custom (Manual Input)": {
                "description": "Design your own airfoil using manual parameter input",
                "camber": 2, "position": 4, "thickness": 12, "naca": "2412",
                "details": "Use the parameter sliders to create your custom airfoil configuration.",
                "technical_details": {
                    "design_philosophy": "User-defined configuration for experimental or educational purposes.",
                    "performance_characteristics": "Performance depends on selected parameters.",
                    "structural_considerations": "Thickness ratio affects structural capability.",
                    "operational_envelope": "Varies based on configuration."
                },
                "application_context": {
                    "primary_use": "Educational and experimental analysis",
                    "aircraft_examples": "Custom designs, student projects",
                    "design_rationale": "Flexible configuration for learning airfoil design principles",
                    "performance_notes": "Allows exploration of parameter effects on aerodynamic performance"
                }
            },
            "Commercial Passenger Aircraft": {
                "description": "Boeing 737, Airbus A320 - Cruise efficiency optimized",
                "camber": 2, "position": 4, "thickness": 15, "naca": "2415",
                "details": "Moderate camber for good lift-to-drag ratio at cruise conditions.",
                "technical_details": {
                    "design_philosophy": "Optimized for fuel efficiency at cruise altitude (35,000-40,000 ft). Moderate camber provides good lift-to-drag ratio while maintaining structural integrity.",
                    "performance_characteristics": "High lift coefficient at cruise angle of attack, low drag coefficient, excellent stall characteristics for passenger safety.",
                    "structural_considerations": "15% thickness provides adequate space for fuel storage, landing gear, and structural strength for commercial operations.",
                    "operational_envelope": "Mach 0.78-0.85 cruise, service ceiling 41,000 ft, typical cruise altitude 35,000-40,000 ft"
                },
                "application_context": {
                    "primary_use": "Commercial passenger transport",
                    "aircraft_examples": "Boeing 737, Airbus A320, Boeing 757, Airbus A321",
                    "design_rationale": "Balance between fuel efficiency, passenger capacity, and operational flexibility. Moderate camber ensures good performance across wide range of weights and altitudes.",
                    "performance_notes": "Optimized for 80% of flight time spent in cruise. Thickness allows internal fuel storage and structural requirements for pressurized cabin."
                }
            },
            "Business Jet": {
                "description": "Citation, Gulfstream - High speed performance",
                "camber": 1, "position": 3, "thickness": 12, "naca": "1312",
                "details": "Low camber and moderate thickness for high-speed cruise efficiency.",
                "technical_details": {
                    "design_philosophy": "Designed for high-speed cruise performance at high altitude. Lower camber reduces drag at high Mach numbers.",
                    "performance_characteristics": "Excellent high-speed characteristics, reduced wave drag onset, good fuel efficiency at Mach 0.80-0.90.",
                    "structural_considerations": "12% thickness provides structural strength while minimizing compressibility effects.",
                    "operational_envelope": "Mach 0.80-0.90 cruise, service ceiling 45,000-51,000 ft, optimized for long-range missions"
                },
                "application_context": {
                    "primary_use": "Executive transport and long-range business travel",
                    "aircraft_examples": "Cessna Citation series, Gulfstream G450/G550, Bombardier Global Express",
                    "design_rationale": "Time-sensitive travel requires high cruise speeds. Lower camber allows higher Mach numbers before drag rise. Forward camber position improves pitching moment characteristics.",
                    "performance_notes": "Designed for minimum trip time rather than maximum fuel efficiency. Higher operating altitudes reduce air traffic conflicts."
                }
            },
            "Military Fighter Aircraft": {
                "description": "F-16, F/A-18 - High speed maneuverability",
                "camber": 0, "position": 4, "thickness": 9, "naca": "0009",
                "details": "Symmetric airfoil with thin section for minimal drag at high speeds.",
                "technical_details": {
                    "design_philosophy": "Symmetric design provides identical performance inverted. Thin section minimizes drag and delays shock formation at supersonic speeds.",
                    "performance_characteristics": "Zero pitching moment coefficient, excellent roll rate, minimal drag at high Mach numbers, suitable for high-g maneuvers.",
                    "structural_considerations": "9% thickness provides minimum practical structure for high-stress combat maneuvers while minimizing drag.",
                    "operational_envelope": "Mach 0.8-2.0+, service ceiling 50,000+ ft, optimized for air-to-air combat and strike missions"
                },
                "application_context": {
                    "primary_use": "Air superiority and multi-role combat operations",
                    "aircraft_examples": "F-16 Fighting Falcon, F/A-18 Hornet, F-22 Raptor wing sections",
                    "design_rationale": "Combat requires inverted flight capability and minimum drag. Symmetric airfoil provides neutral stability for maximum maneuverability. Thin section essential for supersonic performance.",
                    "performance_notes": "Peak suction around -7.762 typical for symmetric sections. Zero camber eliminates asymmetric handling between upright and inverted flight."
                }
            },
            "General Aviation Training": {
                "description": "Cessna 172, Piper Cherokee - Stable flight characteristics",
                "camber": 2, "position": 4, "thickness": 12, "naca": "2412",
                "details": "Classic general aviation airfoil with good stall characteristics.",
                "technical_details": {
                    "design_philosophy": "Designed for benign stall characteristics and stability. Moderate camber provides good low-speed lift for training operations.",
                    "performance_characteristics": "Gentle stall progression, good low-speed handling, forgiving flight characteristics, adequate cruise performance.",
                    "structural_considerations": "12% thickness allows simple, cost-effective construction while providing adequate structural strength.",
                    "operational_envelope": "Mach 0.15-0.45, service ceiling 10,000-14,000 ft, optimized for flight training and recreational flying"
                },
                "application_context": {
                    "primary_use": "Flight training and recreational aviation",
                    "aircraft_examples": "Cessna 172, Piper Cherokee, Beechcraft Musketeer, Diamond DA40",
                    "design_rationale": "Student pilots require predictable, forgiving aircraft. NACA 2412 provides excellent stall warning and recovery characteristics. Proven design with decades of safe operation.",
                    "performance_notes": "Moderate performance optimized for safety and training effectiveness rather than maximum efficiency."
                }
            },
            "Cargo Transport Aircraft": {
                "description": "C-130, Boeing 747F - Heavy load capability",
                "camber": 4, "position": 4, "thickness": 18, "naca": "4418",
                "details": "High camber for maximum lift capability and structural strength.",
                "technical_details": {
                    "design_philosophy": "Maximum lift coefficient for heavy cargo operations. High thickness accommodates structural loads and large internal volume.",
                    "performance_characteristics": "Very high maximum lift coefficient, excellent short-field performance, robust structure for heavy loads.",
                    "structural_considerations": "18% thickness provides maximum structural depth for heavy cargo loads and large internal fuel capacity.",
                    "operational_envelope": "Mach 0.3-0.75, emphasis on payload capacity and short-field performance over cruise speed"
                },
                "application_context": {
                    "primary_use": "Heavy cargo transport and logistics operations",
                    "aircraft_examples": "C-130 Hercules, Boeing 747F, Antonov An-124, Lockheed C-5 Galaxy",
                    "design_rationale": "Cargo operations prioritize payload over speed. High camber maximizes lift for heavy loads. Thick sections provide structural strength and internal volume for cargo and fuel.",
                    "performance_notes": "Peak suction around -9.177 enables high lift coefficients. Design trades cruise efficiency for maximum cargo capability."
                }
            },
            "Wind Turbine Blade": {
                "description": "Wind energy applications - Maximum power extraction",
                "camber": 5, "position": 4, "thickness": 21, "naca": "5421",
                "details": "High camber and thickness for maximum lift coefficient and structural strength.",
                "technical_details": {
                    "design_philosophy": "Designed for maximum power extraction from wind. High camber maximizes lift coefficient across wide angle of attack range.",
                    "performance_characteristics": "Very high lift coefficient, good performance at low Reynolds numbers, robust stall characteristics.",
                    "structural_considerations": "21% thickness provides structural strength for large blade spans and fatigue resistance.",
                    "operational_envelope": "Low speed operation (wind speeds 3-25 m/s), variable angle of attack, 20-30 year service life"
                },
                "application_context": {
                    "primary_use": "Wind energy generation",
                    "aircraft_examples": "Modern wind turbine blades (Vestas, GE, Siemens)",
                    "design_rationale": "Maximum energy extraction requires high lift coefficients. Thick sections provide structural integrity for large rotors. High camber optimizes power output.",
                    "performance_notes": "Designed for maximum Cp (power coefficient) rather than L/D ratio. Structural requirements often override pure aerodynamic optimization."
                }
            },
            "Agricultural Aircraft": {
                "description": "Crop dusters, spray aircraft - Low altitude maneuverability",
                "camber": 3, "position": 4, "thickness": 15, "naca": "3415",
                "details": "Moderate camber for good low-speed lift and control at high angles of attack.",
                "technical_details": {
                    "design_philosophy": "Optimized for low-altitude, slow-speed operations with excellent control authority at high angles of attack. Designed for precise maneuvering around obstacles and terrain following.",
                    "performance_characteristics": "Good low-speed handling, stable at high angles of attack, adequate stall warning, robust performance with external loads.",
                    "structural_considerations": "15% thickness provides structural strength for spray equipment loads and rough field operations.",
                    "operational_envelope": "Mach 0.1-0.3, low altitude operations 10-500 ft AGL, optimized for precision agriculture and aerial application"
                },
                "application_context": {
                    "primary_use": "Agricultural aviation and aerial application",
                    "aircraft_examples": "Air Tractor AT series, Cessna AgWagon, Piper Pawnee, Grumman AgCat",
                    "design_rationale": "Agricultural operations require precise low-speed control, good visibility, and ability to operate from rough airstrips. Moderate camber provides necessary lift at low speeds.",
                    "performance_notes": "Optimized for load-carrying capability and precise control rather than speed. Must handle varying weight distributions from spray loads."
                }
            },
            "Glider Aircraft": {
                "description": "Sailplanes, gliders - Maximum aerodynamic efficiency",
                "camber": 2, "position": 5, "thickness": 12, "naca": "2512",
                "details": "Low camber with aft position for maximum lift-to-drag ratio and efficiency.",
                "technical_details": {
                    "design_philosophy": "Designed for maximum aerodynamic efficiency and minimal energy loss. Optimized for soaring flight and thermal exploitation with minimal sink rates.",
                    "performance_characteristics": "Very high lift-to-drag ratios (40:1 to 60:1), excellent soaring capability, gentle stall characteristics, efficient at multiple speeds.",
                    "structural_considerations": "12% thickness balances structural requirements with aerodynamic efficiency for long wingspan designs.",
                    "operational_envelope": "Mach 0.1-0.4, optimized for soaring flight, service ceiling limited by pilot/oxygen rather than aircraft performance"
                },
                "application_context": {
                    "primary_use": "Recreational and competitive soaring flight",
                    "aircraft_examples": "Schleicher ASK, Schempp-Hirth Discus, Alexander Schleicher ASW, Rolladen-Schneider LS series",
                    "design_rationale": "Soaring flight requires maximum efficiency to exploit weak thermals and ridge lift. Aft camber position helps maintain efficiency across speed range.",
                    "performance_notes": "Designed for minimum sink rate and maximum glide ratio. Performance optimized for energy conservation rather than speed."
                }
            },
            "Aerobatic Aircraft": {
                "description": "Competition aerobatics - Symmetric precision performance",
                "camber": 0, "position": 4, "thickness": 10, "naca": "0010",
                "details": "Symmetric design for identical performance in all flight orientations.",
                "technical_details": {
                    "design_philosophy": "Symmetric design provides identical performance upright, inverted, and knife-edge. Optimized for precision aerobatic maneuvers and competition flying.",
                    "performance_characteristics": "Identical lift and moment characteristics in all orientations, excellent control response, predictable stall in all attitudes, high roll rates.",
                    "structural_considerations": "10% thickness provides structural strength for high-g aerobatic loads while maintaining clean aerodynamics.",
                    "operational_envelope": "Mach 0.2-0.6, optimized for aerobatic box performance, high-g capability (+10/-10g typical), unlimited category competition"
                },
                "application_context": {
                    "primary_use": "Competitive and recreational aerobatics",
                    "aircraft_examples": "Extra 300, Sukhoi Su-26/29/31, Cap 232, Pitts Special, Edge 540",
                    "design_rationale": "Aerobatic competition requires aircraft that perform identically in all orientations. Symmetric airfoils eliminate performance variations between upright and inverted flight.",
                    "performance_notes": "Zero camber ensures no bias toward upright flight. Designed for precision and repeatability rather than efficiency."
                }
            }
        }
    
    @classmethod
    def get_preset_by_name(cls, name: str):
        """Get specific preset by name"""
        presets = cls.get_presets()
        return presets.get(name, presets["Custom (Manual Input)"])
    
    @classmethod
    def get_preset_names(cls):
        """Get list of available preset names"""
        return list(cls.get_presets().keys())
    
    @classmethod
    def get_preset_categories(cls):
        """Get presets organized by category"""
        presets = cls.get_presets()
        categories = {
            "Civil Aviation": [
                "Commercial Passenger Aircraft",
                "Business Jet", 
                "General Aviation Training"
            ],
            "Military": [
                "Military Fighter Aircraft"
            ],
            "Cargo & Transport": [
                "Cargo Transport Aircraft"
            ],
            "Alternative Applications": [
                "Wind Turbine Blade"
            ],
            "Educational": [
                "Custom (Manual Input)"
            ]
        }
        return categories

class PresetComparisons:
    """Predefined comparison sets for multiple airfoil analysis"""
    
    @staticmethod
    def get_comparison_presets():
        """Get predefined comparison configurations"""
        return {
            "Commercial vs Military": {
                "description": "Compare civilian and military aircraft design philosophies",
                "airfoils": [
                    "Commercial Passenger Aircraft",
                    "Military Fighter Aircraft"
                ],
                "analysis_focus": "Design trade-offs between efficiency and performance",
                "key_metrics": ["Peak suction", "Thickness ratio", "Camber effects"]
            },
            "Speed Comparison": {
                "description": "Compare airfoils optimized for different speed ranges",
                "airfoils": [
                    "General Aviation Training",
                    "Commercial Passenger Aircraft", 
                    "Business Jet",
                    "Military Fighter Aircraft"
                ],
                "analysis_focus": "How speed requirements drive airfoil design",
                "key_metrics": ["Thickness ratio", "Camber", "Critical Mach number effects"]
            },
            "Thickness Study": {
                "description": "Study the effects of thickness ratio on performance",
                "airfoils": [
                    "Military Fighter Aircraft",  # 9%
                    "General Aviation Training",  # 12%
                    "Commercial Passenger Aircraft",  # 15%
                    "Cargo Transport Aircraft"  # 18%
                ],
                "analysis_focus": "Structural vs aerodynamic trade-offs",
                "key_metrics": ["Structural capability", "Drag characteristics", "Internal volume"]
            },
            "Camber Effects": {
                "description": "Analyze the impact of different camber configurations",
                "airfoils": [
                    "Military Fighter Aircraft",  # 0% camber
                    "Business Jet",  # 1% camber
                    "Commercial Passenger Aircraft",  # 2% camber
                    "Cargo Transport Aircraft"  # 4% camber
                ],
                "analysis_focus": "Lift generation vs drag penalty",
                "key_metrics": ["Lift coefficient", "Zero-lift angle", "Pitching moment"]
            },
            "Application Extremes": {
                "description": "Compare extreme design requirements",
                "airfoils": [
                    "Military Fighter Aircraft",
                    "Cargo Transport Aircraft", 
                    "Wind Turbine Blade"
                ],
                "analysis_focus": "How extreme requirements shape airfoil design",
                "key_metrics": ["Performance envelope", "Design constraints", "Operational requirements"]
            }
        }
    
    @staticmethod
    def get_educational_comparisons():
        """Get educational comparison sets for learning"""
        return {
            "Beginner": {
                "Symmetric vs Cambered": [
                    "Military Fighter Aircraft",
                    "General Aviation Training"
                ]
            },
            "Intermediate": {
                "Commercial Applications": [
                    "Commercial Passenger Aircraft",
                    "Business Jet",
                    "General Aviation Training"
                ]
            },
            "Advanced": {
                "Design Optimization": [
                    "Military Fighter Aircraft",
                    "Commercial Passenger Aircraft",
                    "Cargo Transport Aircraft",
                    "Wind Turbine Blade"
                ]
            }
        }

class PerformanceAnalyzer:
    """Analyzes airfoil performance characteristics"""
    
    @staticmethod
    def get_performance_rating(airfoil_data: 'AirfoilData', aircraft_type: str) -> tuple:
        """Get performance rating based on aircraft type and Cp characteristics"""
        min_cp_value = abs(airfoil_data.min_cp)
        cp_range_value = airfoil_data.cp_range
        
        # Performance thresholds based on aircraft type
        if aircraft_type == "Military Fighter Aircraft":
            if min_cp_value > 6.0:
                return "Excellent", "success", "High suction typical for fighter aircraft - excellent for high-speed maneuverability"
            elif min_cp_value > 4.0:
                return "Good", "info", "Adequate suction for fighter aircraft applications"
            else:
                return "Needs Improvement", "warning", "Low suction may limit high-speed performance"
        
        elif aircraft_type == "Commercial Passenger Aircraft":
            if 1.5 < min_cp_value < 3.5:
                return "Optimal", "success", "Ideal suction range for cruise efficiency - balances lift and drag"
            elif 1.0 < min_cp_value < 4.0:
                return "Good", "info", "Within acceptable range for commercial aircraft"
            else:
                return "Acceptable", "warning", "Outside optimal range but operational"
        
        elif aircraft_type == "Cargo Transport Aircraft":
            if min_cp_value > PERF_CFG.HIGH_SUCTION_THRESHOLD:
                return "Excellent", "success", "High suction ideal for heavy cargo operations - maximum lift capability"
            elif min_cp_value > PERF_CFG.MODERATE_SUCTION_THRESHOLD:
                return "Good", "info", "Adequate for cargo transport requirements"
            else:
                return "Limited", "warning", "May limit payload capacity"
        
        elif aircraft_type == "Wind Turbine Blade":
            if min_cp_value > 5.0:
                return "Excellent", "success", "Very high suction maximizes power extraction"
            elif min_cp_value > 3.0:
                return "Good", "info", "Good power extraction capability"
            else:
                return "Limited", "warning", "Lower power extraction potential"
        
        else:  # Other aircraft types
            if min_cp_value > PERF_CFG.MODERATE_SUCTION_THRESHOLD:
                return "Good", "success", f"Appropriate suction levels for {aircraft_type.lower()}"
            else:
                return "Adequate", "info", f"Within operational range for {aircraft_type.lower()}"
    
    @staticmethod
    def get_suction_classification(cp_value: float) -> str:
        """Classify suction strength based on Cp value"""
        abs_cp = abs(cp_value)
        if abs_cp > PERF_CFG.HIGH_SUCTION_THRESHOLD:
            return "Very Strong"
        elif abs_cp > PERF_CFG.MODERATE_SUCTION_THRESHOLD:
            return "Strong"
        elif abs_cp > PERF_CFG.MILD_SUCTION_THRESHOLD:
            return "Moderate"
        else:
            return "Mild"
    
    @staticmethod
    def get_cp_range_classification(cp_range: float) -> str:
        """Classify pressure range"""
        if cp_range > PERF_CFG.HIGH_CP_RANGE_THRESHOLD:
            return "High"
        elif cp_range > PERF_CFG.MODERATE_CP_RANGE_THRESHOLD:
            return "Moderate"
        else:
            return "Low"
    
    @staticmethod
    def compare_airfoils(airfoil_list: List['AirfoilData']) -> dict:
        """Compare multiple airfoils and provide insights"""
        if not airfoil_list:
            return {}
        
        comparison = {
            'count': len(airfoil_list),
            'naca_codes': [a.naca_code for a in airfoil_list],
            'min_cp_values': [a.min_cp for a in airfoil_list],
            'max_cp_values': [a.max_cp for a in airfoil_list],
            'cp_ranges': [a.cp_range for a in airfoil_list],
            'camber_values': [a.camber for a in airfoil_list],
            'thickness_values': [a.thickness for a in airfoil_list]
        }
        
        # Statistical analysis
        comparison['stats'] = {
            'avg_min_cp': np.mean(comparison['min_cp_values']),
            'std_min_cp': np.std(comparison['min_cp_values']),
            'avg_thickness': np.mean(comparison['thickness_values']),
            'avg_camber': np.mean(comparison['camber_values']),
            'best_suction_idx': np.argmin(comparison['min_cp_values']),
            'highest_cp_range_idx': np.argmax(comparison['cp_ranges'])
        }
        
        return comparison

# ============================================================================
# PHASE 5: VISUALIZATION COMPONENTS
# ============================================================================

class AirfoilVisualizer:
    """Handles airfoil visualization and plotting"""
    
    @staticmethod
    def create_airfoil_preview(airfoil_data: 'AirfoilData', selected_preset: str):
        """Create airfoil geometry preview plot"""
        fig, ax = plt.subplots(1, 1, figsize=VIZ_CFG.FIGURE_SIZE_PREVIEW, dpi=VIZ_CFG.FIGURE_DPI)
        
        # Plot airfoil shape
        ax.plot(airfoil_data.x_coords, airfoil_data.y_coords, 'navy', 
                linewidth=VIZ_CFG.LINE_WIDTH_MAIN, 
                label=f'NACA {airfoil_data.naca_code}')
        ax.fill(airfoil_data.x_coords, airfoil_data.y_coords, 
                alpha=VIZ_CFG.FILL_ALPHA, color='lightblue')
        
        # Configure plot
        ax.set_aspect('equal')
        ax.grid(True, alpha=VIZ_CFG.GRID_ALPHA)
        ax.set_title(f'NACA {airfoil_data.naca_code} - {selected_preset}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('x/c', fontsize=12)
        ax.set_ylabel('y/c', fontsize=12)
        ax.set_xlim(-0.05, 1.05)
        
        # Add technical information box
        max_thickness = max(airfoil_data.y_coords) - min(airfoil_data.y_coords)
        info_text = (f'Max t/c: {max_thickness:.3f}\n'
                    f'Camber: {airfoil_data.camber}%\n'
                    f'Thickness: {airfoil_data.thickness}%')
        
        if selected_preset != "Custom (Manual Input)":
            preset_info = AircraftPresets.get_preset_by_name(selected_preset)
            info_text += f'\n\nApplication:\n{preset_info["description"]}'
        
        
        
        return fig
    
    @staticmethod
    def create_cp_analysis_plots(airfoil_data: 'AirfoilData'):
        """Create comprehensive Cp analysis plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=VIZ_CFG.FIGURE_SIZE_SINGLE, 
                                                     dpi=VIZ_CFG.FIGURE_DPI)
        
        # 1. Airfoil geometry
        AirfoilVisualizer._plot_airfoil_geometry(ax1, airfoil_data)
        
        # 2. Complete Cp distribution with peak annotation
        AirfoilVisualizer._plot_cp_distribution_with_annotation(ax2, airfoil_data)
        
        # 3. Upper surface Cp detail
        AirfoilVisualizer._plot_upper_surface_detail(ax3, airfoil_data)
        
        # 4. Lower surface Cp detail
        AirfoilVisualizer._plot_lower_surface_detail(ax4, airfoil_data)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def _plot_airfoil_geometry(ax, airfoil_data: 'AirfoilData'):
        """Plot airfoil geometry subplot"""
        ax.plot(airfoil_data.x_coords, airfoil_data.y_coords, 'navy', 
                linewidth=VIZ_CFG.LINE_WIDTH_MAIN, label='Airfoil Shape')
        ax.fill(airfoil_data.x_coords, airfoil_data.y_coords, 
                alpha=VIZ_CFG.FILL_ALPHA, color='lightblue')
        ax.set_aspect('equal')
        ax.grid(True, alpha=VIZ_CFG.GRID_ALPHA)
        ax.set_title(f'NACA {airfoil_data.naca_code} - Geometry', fontweight='bold')
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        ax.legend()
    
    @staticmethod
    def _plot_cp_distribution_with_annotation(ax, airfoil_data: 'AirfoilData'):
        """Plot Cp distribution with peak suction annotation"""
        # Plot Cp curves
        ax.plot(airfoil_data.x_cp, airfoil_data.cp_upper, 'red', 
                linewidth=VIZ_CFG.LINE_WIDTH_MAIN, 
                marker='o', markersize=VIZ_CFG.MARKER_SIZE_MAIN,
                label='Upper Surface', markerfacecolor='red')
        ax.plot(airfoil_data.x_cp, airfoil_data.cp_lower, 'blue',
                linewidth=VIZ_CFG.LINE_WIDTH_MAIN,
                marker='s', markersize=VIZ_CFG.MARKER_SIZE_MAIN,
                label='Lower Surface', markerfacecolor='blue')
        
        ax.invert_yaxis()
        ax.grid(True, alpha=VIZ_CFG.GRID_ALPHA)
        ax.set_title(f'Predicted Cp Distribution - NACA {airfoil_data.naca_code}',
                    fontweight='bold', pad=30)
        ax.set_xlabel('x/c')
        ax.set_ylabel('Pressure Coefficient (Cp)')
        ax.legend()
        
        # Add peak suction annotation
        AirfoilVisualizer._add_peak_suction_annotation(ax, airfoil_data)
    
    @staticmethod
    def _add_peak_suction_annotation(ax, airfoil_data: 'AirfoilData'):
        """Add peak suction annotation to Cp plot"""
        min_cp_idx = np.argmin(airfoil_data.cp_upper)
        min_cp_value = airfoil_data.cp_upper[min_cp_idx]
        min_cp_x = airfoil_data.x_cp[min_cp_idx]
        
        # Smart annotation positioning
        y_range = ax.get_ylim()
        y_span = y_range[0] - y_range[1]
        
        annotation_x = min_cp_x + 0.2 if min_cp_x < 0.5 else min_cp_x - 0.2
        annotation_y = min_cp_value + y_span * 0.2
        
        ax.annotate(f'Peak Suction\nCp = {min_cp_value:.3f}\nx/c = {min_cp_x:.2f}',
                   xy=(min_cp_x, min_cp_value),
                   xytext=(annotation_x, annotation_y),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=10, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", 
                            alpha=0.8, edgecolor='red'))
    
    @staticmethod
    def _plot_upper_surface_detail(ax, airfoil_data: 'AirfoilData'):
        """Plot upper surface Cp detail"""
        ax.plot(airfoil_data.x_cp, airfoil_data.cp_upper, 'red',
                linewidth=VIZ_CFG.LINE_WIDTH_DETAIL, 
                marker='o', markersize=VIZ_CFG.MARKER_SIZE_DETAIL)
        ax.invert_yaxis()
        ax.grid(True, alpha=VIZ_CFG.GRID_ALPHA)
        ax.set_title('Upper Surface Cp (Detailed)', fontweight='bold')
        ax.set_xlabel('x/c')
        ax.set_ylabel('Cp')
        ax.fill_between(airfoil_data.x_cp, airfoil_data.cp_upper, 
                       alpha=VIZ_CFG.FILL_ALPHA, color='red')
    
    @staticmethod
    def _plot_lower_surface_detail(ax, airfoil_data: 'AirfoilData'):
        """Plot lower surface Cp detail"""
        ax.plot(airfoil_data.x_cp, airfoil_data.cp_lower, 'blue',
                linewidth=VIZ_CFG.LINE_WIDTH_DETAIL,
                marker='s', markersize=VIZ_CFG.MARKER_SIZE_DETAIL)
        ax.invert_yaxis()
        ax.grid(True, alpha=VIZ_CFG.GRID_ALPHA)
        ax.set_title('Lower Surface Cp (Detailed)', fontweight='bold')
        ax.set_xlabel('x/c')
        ax.set_ylabel('Cp')
        ax.fill_between(airfoil_data.x_cp, airfoil_data.cp_lower,
                       alpha=VIZ_CFG.FILL_ALPHA, color='blue')

class ComparisonVisualizer:
    """Handles multiple airfoil comparison visualizations"""
    
    @staticmethod
    def create_comparison_plots(airfoil_data_list: List[dict]):
        """Create comprehensive comparison plots for multiple airfoils"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=VIZ_CFG.FIGURE_SIZE_COMPARISON, 
                                                     dpi=VIZ_CFG.FIGURE_DPI)
        
        # 1. All airfoil geometries overlaid
        ComparisonVisualizer._plot_geometry_comparison(ax1, airfoil_data_list)
        
        # 2. Upper surface Cp comparison
        ComparisonVisualizer._plot_upper_cp_comparison(ax2, airfoil_data_list)
        
        # 3. Lower surface Cp comparison
        ComparisonVisualizer._plot_lower_cp_comparison(ax3, airfoil_data_list)
        
        # 4. Complete Cp comparison
        ComparisonVisualizer._plot_complete_cp_comparison(ax4, airfoil_data_list)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def _plot_geometry_comparison(ax, airfoil_data_list: List[dict]):
        """Plot geometry comparison subplot"""
        ax.set_title('Airfoil Geometry Comparison', fontweight='bold', fontsize=14)
        
        for i, data_dict in enumerate(airfoil_data_list):
            airfoil_obj = data_dict['data']
            config = data_dict['config']
            aircraft_type = config['selected_preset'].split()[0] if config['selected_preset'] != 'Custom (Manual Input)' else 'Custom'
            label = f"NACA {airfoil_obj.naca_code} ({aircraft_type})"
            
            color = VIZ_CFG.AIRFOIL_COLORS[i % len(VIZ_CFG.AIRFOIL_COLORS)]
            linestyle = VIZ_CFG.AIRFOIL_LINE_STYLES[i % len(VIZ_CFG.AIRFOIL_LINE_STYLES)]
            
            ax.plot(airfoil_obj.x_coords, airfoil_obj.y_coords, 
                    color=color, linewidth=VIZ_CFG.LINE_WIDTH_MAIN,
                    linestyle=linestyle, label=label)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=VIZ_CFG.GRID_ALPHA)
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        ax.legend()
        ax.set_xlim(-0.05, 1.05)
    
    @staticmethod
    def _plot_upper_cp_comparison(ax, airfoil_data_list: List[dict]):
        """Plot upper surface Cp comparison"""
        ax.set_title('Upper Surface Cp Comparison', fontweight='bold', fontsize=14)
        
        for i, data_dict in enumerate(airfoil_data_list):
            airfoil_obj = data_dict['data']
            color = VIZ_CFG.AIRFOIL_COLORS[i % len(VIZ_CFG.AIRFOIL_COLORS)]
            linestyle = VIZ_CFG.AIRFOIL_LINE_STYLES[i % len(VIZ_CFG.AIRFOIL_LINE_STYLES)]
            label = f"NACA {airfoil_obj.naca_code}"
            
            ax.plot(airfoil_obj.x_cp, airfoil_obj.cp_upper, 
                    color=color, linewidth=VIZ_CFG.LINE_WIDTH_MAIN,
                    linestyle=linestyle, marker='o', markersize=VIZ_CFG.MARKER_SIZE_DETAIL,
                    label=label, markerfacecolor=color)
        
        ax.invert_yaxis()
        ax.grid(True, alpha=VIZ_CFG.GRID_ALPHA)
        ax.set_xlabel('x/c')
        ax.set_ylabel('Cp')
        ax.legend()
    
    @staticmethod
    def _plot_lower_cp_comparison(ax, airfoil_data_list: List[dict]):
        """Plot lower surface Cp comparison"""
        ax.set_title('Lower Surface Cp Comparison', fontweight='bold', fontsize=14)
        
        for i, data_dict in enumerate(airfoil_data_list):
            airfoil_obj = data_dict['data']
            color = VIZ_CFG.AIRFOIL_COLORS[i % len(VIZ_CFG.AIRFOIL_COLORS)]
            linestyle = VIZ_CFG.AIRFOIL_LINE_STYLES[i % len(VIZ_CFG.AIRFOIL_LINE_STYLES)]
            label = f"NACA {airfoil_obj.naca_code}"
            
            ax.plot(airfoil_obj.x_cp, airfoil_obj.cp_lower,
                    color=color, linewidth=VIZ_CFG.LINE_WIDTH_MAIN,
                    linestyle=linestyle, marker='s', markersize=VIZ_CFG.MARKER_SIZE_DETAIL,
                    label=label, markerfacecolor=color)
        
        ax.invert_yaxis()
        ax.grid(True, alpha=VIZ_CFG.GRID_ALPHA)
        ax.set_xlabel('x/c')
        ax.set_ylabel('Cp')
        ax.legend()
    
    @staticmethod
    def _plot_complete_cp_comparison(ax, airfoil_data_list: List[dict]):
        """Plot complete Cp comparison"""
        ax.set_title('Complete Cp Distribution Comparison', fontweight='bold', fontsize=14)
        
        for i, data_dict in enumerate(airfoil_data_list):
            airfoil_obj = data_dict['data']
            color = VIZ_CFG.AIRFOIL_COLORS[i % len(VIZ_CFG.AIRFOIL_COLORS)]
            
            ax.plot(airfoil_obj.x_cp, airfoil_obj.cp_upper,
                    color=color, linewidth=VIZ_CFG.LINE_WIDTH_DETAIL,
                    linestyle='-', alpha=0.8, label=f"NACA {airfoil_obj.naca_code} (Upper)")
            ax.plot(airfoil_obj.x_cp, airfoil_obj.cp_lower,
                    color=color, linewidth=VIZ_CFG.LINE_WIDTH_DETAIL,
                    linestyle='--', alpha=0.8, label=f"NACA {airfoil_obj.naca_code} (Lower)")
        
        ax.invert_yaxis()
        ax.grid(True, alpha=VIZ_CFG.GRID_ALPHA)
        ax.set_xlabel('x/c')
        ax.set_ylabel('Cp')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

class PerformanceVisualizer:
    """Creates performance analysis visualizations"""
    
    @staticmethod
    def create_performance_comparison_chart(airfoil_data_list: List[dict]):
        """Create performance metrics comparison chart"""
        # Extract data for comparison
        naca_codes = []
        min_cps = []
        max_cps = []
        cp_ranges = []
        colors = []
        
        for i, data_dict in enumerate(airfoil_data_list):
            airfoil_obj = data_dict['data']
            naca_codes.append(f"NACA {airfoil_obj.naca_code}")
            min_cps.append(airfoil_obj.min_cp)
            max_cps.append(airfoil_obj.max_cp)
            cp_ranges.append(airfoil_obj.cp_range)
            colors.append(VIZ_CFG.AIRFOIL_COLORS[i % len(VIZ_CFG.AIRFOIL_COLORS)])
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), dpi=VIZ_CFG.FIGURE_DPI)
        
        # Peak suction comparison
        bars1 = ax1.bar(naca_codes, [abs(cp) for cp in min_cps], color=colors, alpha=0.7)
        ax1.set_title('Peak Suction Comparison', fontweight='bold')
        ax1.set_ylabel('|Min Cp| (Peak Suction)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=VIZ_CFG.GRID_ALPHA)
        
        # Add value labels on bars
        for bar, value in zip(bars1, min_cps):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Pressure range comparison
        bars2 = ax2.bar(naca_codes, cp_ranges, color=colors, alpha=0.7)
        ax2.set_title('Pressure Range Comparison', fontweight='bold')
        ax2.set_ylabel('Cp Range')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=VIZ_CFG.GRID_ALPHA)
        
        for bar, value in zip(bars2, cp_ranges):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Scatter plot: Peak Suction vs Range
        scatter = ax3.scatter([abs(cp) for cp in min_cps], cp_ranges, 
                            c=colors, s=100, alpha=0.7, edgecolors='black')
        ax3.set_title('Peak Suction vs Pressure Range', fontweight='bold')
        ax3.set_xlabel('|Min Cp| (Peak Suction)')
        ax3.set_ylabel('Cp Range')
        ax3.grid(True, alpha=VIZ_CFG.GRID_ALPHA)
        
        # Add labels to scatter points
        for i, naca in enumerate(naca_codes):
            ax3.annotate(naca.split()[1], (abs(min_cps[i]), cp_ranges[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.tight_layout()
        return fig

class PlotUtilities:
    """Utility functions for plot management and styling"""
    
    @staticmethod
    def apply_style_theme():
        """Apply consistent styling theme to matplotlib plots"""
        plt.style.use('default')  # Reset to default first
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'grid.alpha': VIZ_CFG.GRID_ALPHA,
            'lines.linewidth': VIZ_CFG.LINE_WIDTH_DETAIL
        })
    
    @staticmethod
    def save_plot_for_pdf(fig, filename: str, dpi: int = 300):
        """Save plot as high-quality image for PDF inclusion"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        return buffer
    
    @staticmethod
    def cleanup_figures():
        """Clean up matplotlib figures to prevent memory leaks"""
        plt.close('all')
    
    @staticmethod
    def create_legend_only_plot(labels: List[str], colors: List[str], 
                              line_styles: List[str] = None):
        """Create a standalone legend plot for multi-page PDF reports"""
        if line_styles is None:
            line_styles = ['-'] * len(labels)
        
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Create legend entries
        legend_elements = []
        for label, color, style in zip(labels, colors, line_styles):
            from matplotlib.lines import Line2D
            legend_elements.append(Line2D([0], [0], color=color, lw=3, 
                                        linestyle=style, label=label))
        
        ax.legend(handles=legend_elements, loc='center', ncol=min(len(labels), 4),
                 frameon=False, fontsize=12)
        
        return fig

# ============================================================================
# PHASE 6: COMPLETE PDF REPORT GENERATION
# ============================================================================

class PDFReportGenerator:
    """Comprehensive PDF report generation for airfoil analysis"""
    
    def __init__(self):
        self.styles = self._create_styles()
        self.available = REPORTLAB_AVAILABLE
    
    def _create_styles(self):
        """Create custom styles for PDF reports"""
        if not REPORTLAB_AVAILABLE:
            return None
            
        styles = getSampleStyleSheet()
        
        # Custom title style
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Title'],
            fontSize=REPORT_CFG.TITLE_FONT_SIZE,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=REPORT_CFG.HEADER_COLOR
        ))
        
        # Custom header style
        styles.add(ParagraphStyle(
            name='CustomHeader',
            parent=styles['Heading1'],
            fontSize=REPORT_CFG.HEADER_FONT_SIZE,
            spaceAfter=12,
            spaceBefore=20,
            textColor=REPORT_CFG.HEADER_COLOR
        ))
        
        # Custom body style
        styles.add(ParagraphStyle(
            name='CustomBody',
            parent=styles['Normal'],
            fontSize=REPORT_CFG.BODY_FONT_SIZE,
            spaceAfter=6,
            alignment=TA_LEFT
        ))
        
        # Custom caption style
        styles.add(ParagraphStyle(
            name='CustomCaption',
            parent=styles['Normal'],
            fontSize=REPORT_CFG.CAPTION_FONT_SIZE,
            spaceAfter=6,
            alignment=TA_CENTER,
            textColor=colors.grey
        ))
        
        return styles
    
    def generate_single_airfoil_report(self, airfoil_data: 'AirfoilData', 
                                     selected_preset: str, 
                                     analysis_plots: List[plt.Figure]) -> io.BytesIO:
        """Generate comprehensive PDF report for single airfoil analysis"""
        if not self.available:
            raise RuntimeError("PDF generation requires reportlab package")
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=REPORT_CFG.PAGE_SIZE,
            rightMargin=REPORT_CFG.MARGIN,
            leftMargin=REPORT_CFG.MARGIN,
            topMargin=REPORT_CFG.MARGIN,
            bottomMargin=REPORT_CFG.MARGIN
        )
        
        # Build report content
        story = []
        
        # Title page
        story.extend(self._create_title_page(airfoil_data, selected_preset))
        
        # Executive summary
        story.extend(self._create_executive_summary(airfoil_data, selected_preset))
        
        # Technical specifications
        story.extend(self._create_technical_specifications(airfoil_data, selected_preset))
        
        # Analysis results
        story.extend(self._create_analysis_results(airfoil_data, analysis_plots))
        
        # Performance assessment
        story.extend(self._create_performance_assessment(airfoil_data, selected_preset))
        
        # Engineering insights
        story.extend(self._create_engineering_insights(airfoil_data, selected_preset))
        
        # Appendix
        story.extend(self._create_appendix(airfoil_data))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def generate_comparison_report(self, airfoil_data_list: List[dict], 
                                 comparison_plots: List[plt.Figure]) -> io.BytesIO:
        """Generate comprehensive PDF report for multiple airfoil comparison"""
        if not self.available:
            raise RuntimeError("PDF generation requires reportlab package")
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=REPORT_CFG.PAGE_SIZE,
            rightMargin=REPORT_CFG.MARGIN,
            leftMargin=REPORT_CFG.MARGIN,
            topMargin=REPORT_CFG.MARGIN,
            bottomMargin=REPORT_CFG.MARGIN
        )
        
        # Build report content
        story = []
        
        # Title page
        story.extend(self._create_comparison_title_page(airfoil_data_list))
        
        # Executive summary
        story.extend(self._create_comparison_executive_summary(airfoil_data_list))
        
        # Individual airfoil specifications
        story.extend(self._create_individual_specifications(airfoil_data_list))
        
        # Comparative analysis
        story.extend(self._create_comparative_analysis(airfoil_data_list, comparison_plots))
        
        # Performance comparison
        story.extend(self._create_performance_comparison(airfoil_data_list))
        
        # Engineering recommendations
        story.extend(self._create_engineering_recommendations(airfoil_data_list))
        
        # Detailed data appendix
        story.extend(self._create_comparison_appendix(airfoil_data_list))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def _create_title_page(self, airfoil_data: 'AirfoilData', selected_preset: str):
        """Create title page for single airfoil report"""
        story = []
        
        # Main title
        title = f"NACA {airfoil_data.naca_code} Airfoil Analysis Report"
        story.append(Paragraph(title, self.styles['CustomTitle']))
        story.append(Spacer(1, 20))
        
        # Subtitle
        subtitle = f"Aircraft Application: {selected_preset}"
        story.append(Paragraph(subtitle, self.styles['CustomHeader']))
        story.append(Spacer(1, 30))
        
        # Analysis details table
        analysis_data = [
            ['Analysis Parameter', 'Value'],
            ['NACA Code', airfoil_data.naca_code],
            ['Camber', f"{airfoil_data.camber}%"],
            ['Camber Position', f"{airfoil_data.position_ratio*100:.0f}%" if airfoil_data.camber > 0 else "N/A"],
            ['Thickness', f"{airfoil_data.thickness}%"],
            ['Analysis Date', airfoil_data.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')],
            ['Analysis Method', 'Neural Network Prediction']
        ]
        
        table = Table(analysis_data, colWidths=[2.5*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), REPORT_CFG.TABLE_HEADER_COLOR),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(table)
        story.append(Spacer(1, 30))
        
        # Report purpose
        purpose_text = """
        This report provides a comprehensive aerodynamic analysis of the specified NACA 4-digit airfoil 
        using advanced neural network prediction methods. The analysis includes pressure coefficient 
        distribution, performance characteristics, and engineering insights relevant to the specified 
        aircraft application.
        """
        story.append(Paragraph(purpose_text, self.styles['CustomBody']))
        
        story.append(PageBreak())
        return story
    
    def _create_executive_summary(self, airfoil_data: 'AirfoilData', selected_preset: str):
        """Create executive summary section"""
        story = []
        
        story.append(Paragraph("Executive Summary", self.styles['CustomHeader']))
        
        # Get performance rating
        performance_rating, _, performance_note = PerformanceAnalyzer.get_performance_rating(
            airfoil_data, selected_preset
        )
        
        # Key findings
        summary_text = f"""
        <b>Airfoil:</b> NACA {airfoil_data.naca_code}<br/>
        <b>Application:</b> {selected_preset}<br/>
        <b>Performance Rating:</b> {performance_rating}<br/>
        <b>Peak Suction (Min Cp):</b> {airfoil_data.min_cp:.3f}<br/>
        <b>Pressure Range:</b> {airfoil_data.cp_range:.3f}<br/><br/>
        
        <b>Key Findings:</b><br/>
        • {performance_note}<br/>
        • Suction classification: {PerformanceAnalyzer.get_suction_classification(airfoil_data.min_cp)}<br/>
        • Pressure range classification: {PerformanceAnalyzer.get_cp_range_classification(airfoil_data.cp_range)}<br/>
        """
        
        if airfoil_data.is_symmetric:
            summary_text += "• Symmetric airfoil design provides identical performance when inverted<br/>"
        else:
            summary_text += f"• Cambered design with maximum camber at {airfoil_data.position_ratio*100:.0f}% chord<br/>"
        
        story.append(Paragraph(summary_text, self.styles['CustomBody']))
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_technical_specifications(self, airfoil_data: 'AirfoilData', selected_preset: str):
        """Create technical specifications section"""
        story = []
        
        story.append(Paragraph("Technical Specifications", self.styles['CustomHeader']))
        
        # NACA parameters table
        naca_data = [
            ['Parameter', 'Value', 'Description'],
            ['First Digit (M)', str(airfoil_data.camber), 'Maximum camber as % of chord'],
            ['Second Digit (P)', str(airfoil_data.position) if airfoil_data.camber > 0 else 'N/A', 
             'Position of maximum camber (tenths of chord)'],
            ['Last Two Digits (XX)', f"{airfoil_data.thickness:02d}", 'Maximum thickness as % of chord'],
        ]
        
        naca_table = Table(naca_data, colWidths=[1.5*inch, 1*inch, 3*inch])
        naca_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), REPORT_CFG.TABLE_HEADER_COLOR),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        story.append(naca_table)
        story.append(Spacer(1, 20))
        
        # Application context if not custom
        if selected_preset != "Custom (Manual Input)":
            preset_info = AircraftPresets.get_preset_by_name(selected_preset)
            
            story.append(Paragraph("Application Context", self.styles['CustomHeader']))
            
            context_text = f"""
            <b>Primary Use:</b> {preset_info['application_context']['primary_use']}<br/>
            <b>Aircraft Examples:</b> {preset_info['application_context']['aircraft_examples']}<br/>
            <b>Design Rationale:</b> {preset_info['application_context']['design_rationale']}<br/><br/>
            
            <b>Technical Details:</b><br/>
            {preset_info['technical_details']['design_philosophy']}<br/><br/>
            
            <b>Performance Characteristics:</b><br/>
            {preset_info['technical_details']['performance_characteristics']}<br/><br/>
            
            <b>Operational Envelope:</b><br/>
            {preset_info['technical_details']['operational_envelope']}
            """
            
            story.append(Paragraph(context_text, self.styles['CustomBody']))
        
        return story
    
    def _create_analysis_results(self, airfoil_data: 'AirfoilData', analysis_plots: List[plt.Figure]):
        """Create analysis results section with embedded plots"""
        story = []
        
        story.append(Paragraph("Analysis Results", self.styles['CustomHeader']))
        
        # Pressure coefficient summary
        results_text = f"""
        The neural network analysis of NACA {airfoil_data.naca_code} reveals the following pressure 
        coefficient characteristics:<br/><br/>
        
        <b>Upper Surface Analysis:</b><br/>
        • Minimum Cp (Peak Suction): {airfoil_data.cp_upper.min():.4f}<br/>
        • Mean Cp: {airfoil_data.cp_upper.mean():.4f}<br/>
        • Peak suction location: x/c = {airfoil_data.x_cp[np.argmin(airfoil_data.cp_upper)]:.3f}<br/><br/>
        
        <b>Lower Surface Analysis:</b><br/>
        • Maximum Cp: {airfoil_data.cp_lower.max():.4f}<br/>
        • Mean Cp: {airfoil_data.cp_lower.mean():.4f}<br/><br/>
        
        <b>Overall Characteristics:</b><br/>
        • Total pressure range: {airfoil_data.cp_range:.4f}<br/>
        • Overall mean Cp: {airfoil_data.cp_distribution.mean():.4f}
        """
        
        story.append(Paragraph(results_text, self.styles['CustomBody']))
        story.append(Spacer(1, 20))
        
        # Embed analysis plots
        for i, fig in enumerate(analysis_plots):
            plot_buffer = PlotUtilities.save_plot_for_pdf(fig, f"analysis_plot_{i}")
            img = Image(plot_buffer, width=6*inch, height=4.5*inch)
            story.append(img)
            story.append(Paragraph(f"Figure {i+1}: Airfoil analysis visualization", 
                                 self.styles['CustomCaption']))
            story.append(Spacer(1, 15))
        
        return story
    
    def _create_performance_assessment(self, airfoil_data: 'AirfoilData', selected_preset: str):
        """Create performance assessment section"""
        story = []
        
        story.append(Paragraph("Performance Assessment", self.styles['CustomHeader']))
        
        # Get performance metrics
        performance_rating, status, performance_note = PerformanceAnalyzer.get_performance_rating(
            airfoil_data, selected_preset
        )
        
        assessment_text = f"""
        <b>Overall Performance Rating:</b> {performance_rating}<br/><br/>
        
        <b>Assessment Details:</b><br/>
        {performance_note}<br/><br/>
        
        <b>Aerodynamic Characteristics:</b><br/>
        • Suction Strength: {PerformanceAnalyzer.get_suction_classification(airfoil_data.min_cp)}<br/>
        • Pressure Range: {PerformanceAnalyzer.get_cp_range_classification(airfoil_data.cp_range)}<br/>
        """
        
        if airfoil_data.is_symmetric:
            assessment_text += """
            • Symmetric Design Benefits:<br/>
              - Identical performance when inverted<br/>
              - Zero pitching moment at zero lift<br/>
              - Excellent for aerobatic applications<br/>
            """
        else:
            assessment_text += f"""
            • Cambered Design Benefits:<br/>
              - Enhanced lift generation at positive angles of attack<br/>
              - Improved efficiency for normal flight operations<br/>
              - Camber optimized for {selected_preset.lower()}<br/>
            """
        
        story.append(Paragraph(assessment_text, self.styles['CustomBody']))
        
        return story
    
    def _create_engineering_insights(self, airfoil_data: 'AirfoilData', selected_preset: str):
        """Create engineering insights and recommendations section"""
        story = []
        
        story.append(Paragraph("Engineering Insights", self.styles['CustomHeader']))
        
        # Application-specific insights
        min_cp_abs = abs(airfoil_data.min_cp)
        
        if selected_preset == "Commercial Passenger Aircraft":
            insights_text = f"""
            <b>Commercial Aviation Assessment:</b><br/>
            With a peak suction of {airfoil_data.min_cp:.3f}, this airfoil configuration provides 
            {"excellent" if 1.0 < min_cp_abs < 3.0 else "adequate"} performance for commercial operations.<br/><br/>
            
            <b>Operational Considerations:</b><br/>
            • Fuel efficiency: {"Optimized" if 1.5 < min_cp_abs < 2.5 else "Acceptable"} for cruise conditions<br/>
            • Passenger comfort: Stable pressure distribution reduces cabin turbulence<br/>
            • Maintenance: {airfoil_data.thickness}% thickness provides adequate structural margins<br/>
            • Safety: Predictable stall characteristics enhance operational safety<br/><br/>
            
            <b>Recommendations:</b><br/>
            • Suitable for cruise altitudes 35,000-40,000 ft<br/>
            • Optimal for Mach 0.78-0.85 cruise speeds<br/>
            • Consider high-lift devices for takeoff/landing performance
            """
        
        elif selected_preset == "Military Fighter Aircraft":
            insights_text = f"""
            <b>Military Aircraft Assessment:</b><br/>
            The symmetric NACA {airfoil_data.naca_code} design provides 
            {"excellent" if min_cp_abs > 6.0 else "good"} characteristics for combat operations.<br/><br/>
            
            <b>Combat Considerations:</b><br/>
            • Maneuverability: Identical performance upright and inverted<br/>
            • High-speed capability: {airfoil_data.thickness}% thickness minimizes wave drag<br/>
            • Structural efficiency: Thin section optimized for high-g loads<br/>
            • Roll rate: Symmetric design enables rapid roll maneuvers<br/><br/>
            
            <b>Recommendations:</b><br/>
            • Suitable for Mach 0.8-2.0+ operations<br/>
            • Excellent for air-to-air combat scenarios<br/>
            • Consider leading edge devices for enhanced maneuverability
            """
        
        elif selected_preset == "Cargo Transport Aircraft":
            insights_text = f"""
            <b>Cargo Transport Assessment:</b><br/>
            The high-camber NACA {airfoil_data.naca_code} maximizes lift capability for heavy loads.<br/><br/>
            
            <b>Cargo Operations:</b><br/>
            • Payload capacity: {airfoil_data.camber}% camber maximizes lift coefficient<br/>
            • Short-field performance: High camber enables steep approach angles<br/>
            • Structural capability: {airfoil_data.thickness}% thickness supports heavy loads<br/>
            • Operational flexibility: Robust performance across weight ranges<br/><br/>
            
            <b>Recommendations:</b><br/>
            • Optimize for maximum payload rather than cruise speed<br/>
            • Suitable for short and rough runway operations<br/>
            • Consider high-lift systems for enhanced short-field capability
            """
        
        else:
            insights_text = f"""
            <b>General Assessment:</b><br/>
            The NACA {airfoil_data.naca_code} configuration provides balanced performance characteristics
            suitable for the specified application.<br/><br/>
            
            <b>Key Characteristics:</b><br/>
            • Moderate performance across operational envelope<br/>
            • {airfoil_data.thickness}% thickness provides structural adequacy<br/>
            • {"Symmetric" if airfoil_data.is_symmetric else "Cambered"} design optimized for intended use<br/>
            • Pressure distribution indicates {"good" if min_cp_abs > 2.0 else "adequate"} aerodynamic efficiency
            """
        
        story.append(Paragraph(insights_text, self.styles['CustomBody']))
        
        return story
    
    def _create_appendix(self, airfoil_data: 'AirfoilData'):
        """Create appendix with detailed data"""
        story = []
        
        story.append(PageBreak())
        story.append(Paragraph("Appendix: Detailed Data", self.styles['CustomHeader']))
        
        # Pressure coefficient data table (sample)
        story.append(Paragraph("Pressure Coefficient Data (Sample)", self.styles['CustomBody']))
        
        # Create data table with every 10th point to fit on page
        step = max(1, len(airfoil_data.x_cp) // 20)  # Show ~20 data points
        cp_data = [['x/c', 'Cp Upper', 'Cp Lower']]
        
        for i in range(0, len(airfoil_data.x_cp), step):
            cp_data.append([
                f"{airfoil_data.x_cp[i]:.3f}",
                f"{airfoil_data.cp_upper[i]:.4f}",
                f"{airfoil_data.cp_lower[i]:.4f}"
            ])
        
        cp_table = Table(cp_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
        cp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), REPORT_CFG.TABLE_HEADER_COLOR),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(cp_table)
        story.append(Spacer(1, 20))
        
        # Analysis methodology
        story.append(Paragraph("Analysis Methodology", self.styles['CustomBody']))
        methodology_text = """
        This analysis was performed using a trained neural network model that predicts pressure 
        coefficient distributions for NACA 4-digit airfoils. The model was trained on a comprehensive 
        database of computational fluid dynamics (CFD) results for various NACA configurations.
        
        Key assumptions:
        • Inviscid flow conditions
        • Zero angle of attack
        • Subsonic flow regime (Mach < 0.8)
        • Standard atmospheric conditions
        • Two-dimensional analysis
        
        The neural network input consists of 400 scaled airfoil coordinates (200 x,y pairs) and 
        outputs 200 pressure coefficient values (100 upper surface + 100 lower surface points).
        """
        
        story.append(Paragraph(methodology_text, self.styles['CustomBody']))
        
        return story
    
    def _create_comparison_title_page(self, airfoil_data_list: List[dict]):
        """Create title page for comparison report"""
        story = []
        
        # Main title
        title = "Multi-Airfoil Comparative Analysis Report"
        story.append(Paragraph(title, self.styles['CustomTitle']))
        story.append(Spacer(1, 20))
        
        # Subtitle
        subtitle = f"Comparative Study of {len(airfoil_data_list)} NACA Airfoils"
        story.append(Paragraph(subtitle, self.styles['CustomHeader']))
        story.append(Spacer(1, 30))
        
        # Airfoils being compared
        comparison_data = [['Airfoil', 'NACA Code', 'Application', 'Camber', 'Thickness']]
        
        for i, data_dict in enumerate(airfoil_data_list):
            airfoil_obj = data_dict['data']
            config = data_dict['config']
            application = config['selected_preset'].split()[0] if config['selected_preset'] != "Custom (Manual Input)" else "Custom"
            
            comparison_data.append([
                f"Airfoil {i+1}",
                airfoil_obj.naca_code,
                application,
                f"{airfoil_obj.camber}%",
                f"{airfoil_obj.thickness}%"
            ])
        
        comparison_table = Table(comparison_data, colWidths=[1*inch, 1*inch, 1.5*inch, 1*inch, 1*inch])
        comparison_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), REPORT_CFG.TABLE_HEADER_COLOR),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(comparison_table)
        story.append(Spacer(1, 30))
        
        # Analysis timestamp
        analysis_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        story.append(Paragraph(f"Analysis Date: {analysis_time}", self.styles['CustomBody']))
        story.append(Paragraph("Analysis Method: Neural Network Prediction", self.styles['CustomBody']))
        
        story.append(PageBreak())
        return story
    
    def _create_comparison_executive_summary(self, airfoil_data_list: List[dict]):
        """Create executive summary for comparison report"""
        story = []
        
        story.append(Paragraph("Executive Summary", self.styles['CustomHeader']))
        
        # Extract key metrics
        naca_codes = [data['data'].naca_code for data in airfoil_data_list]
        min_cps = [data['data'].min_cp for data in airfoil_data_list]
        cp_ranges = [data['data'].cp_range for data in airfoil_data_list]
        
        # Find extremes
        best_suction_idx = np.argmin(min_cps)
        highest_range_idx = np.argmax(cp_ranges)
        
        summary_text = f"""
        This comparative analysis examines {len(airfoil_data_list)} NACA 4-digit airfoils across different 
        applications and design philosophies.<br/><br/>
        
        <b>Key Findings:</b><br/>
        • Airfoils analyzed: {', '.join(naca_codes)}<br/>
        • Peak suction range: {min(min_cps):.3f} to {max(min_cps):.3f}<br/>
        • Pressure range span: {min(cp_ranges):.3f} to {max(cp_ranges):.3f}<br/><br/>
        
        <b>Performance Leaders:</b><br/>
        • Highest suction: NACA {naca_codes[best_suction_idx]} (Cp = {min_cps[best_suction_idx]:.3f})<br/>
        • Largest pressure range: NACA {naca_codes[highest_range_idx]} (Range = {cp_ranges[highest_range_idx]:.3f})<br/><br/>
        
        <b>Design Trade-offs Observed:</b><br/>
        • Thickness vs. drag characteristics<br/>
        • Camber vs. operational versatility<br/>
        • Application-specific optimization strategies
        """
        
        story.append(Paragraph(summary_text, self.styles['CustomBody']))
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_individual_specifications(self, airfoil_data_list: List[dict]):
        """Create individual airfoil specifications section"""
        story = []
        
        story.append(Paragraph("Individual Airfoil Specifications", self.styles['CustomHeader']))
        
        for i, data_dict in enumerate(airfoil_data_list):
            airfoil_obj = data_dict['data']
            config = data_dict['config']
            
            story.append(Paragraph(f"Airfoil {i+1}: NACA {airfoil_obj.naca_code}", 
                                 self.styles['CustomBody']))
            
            spec_text = f"""
            <b>Configuration:</b> {config['selected_preset']}<br/>
            <b>Camber:</b> {airfoil_obj.camber}%<br/>
            <b>Camber Position:</b> {airfoil_obj.position_ratio*100:.0f}% chord<br/>
            <b>Thickness:</b> {airfoil_obj.thickness}%<br/>
            <b>Peak Suction (Min Cp):</b> {airfoil_obj.min_cp:.4f}<br/>
            <b>Pressure Range:</b> {airfoil_obj.cp_range:.4f}<br/>
            """
            
            story.append(Paragraph(spec_text, self.styles['CustomBody']))
            story.append(Spacer(1, 10))
        
        return story
    
    def _create_comparative_analysis(self, airfoil_data_list: List[dict], comparison_plots: List[plt.Figure]):
        """Create comparative analysis section with plots"""
        story = []
        
        story.append(Paragraph("Comparative Analysis", self.styles['CustomHeader']))
        
        # Embed comparison plots
        for i, fig in enumerate(comparison_plots):
            plot_buffer = PlotUtilities.save_plot_for_pdf(fig, f"comparison_plot_{i}")
            img = Image(plot_buffer, width=6*inch, height=4.5*inch)
            story.append(img)
            story.append(Paragraph(f"Figure {i+1}: Comparative analysis visualization", 
                                 self.styles['CustomCaption']))
            story.append(Spacer(1, 15))
        
        return story
    
    def _create_performance_comparison(self, airfoil_data_list: List[dict]):
        """Create performance comparison section"""
        story = []
        
        story.append(Paragraph("Performance Comparison", self.styles['CustomHeader']))
        
        # Create comprehensive comparison table
        perf_data = [['Airfoil', 'NACA', 'Min Cp', 'Max Cp', 'Range', 'Suction Class', 'Range Class']]
        
        for data_dict in airfoil_data_list:
            airfoil_obj = data_dict['data']
            suction_class = PerformanceAnalyzer.get_suction_classification(airfoil_obj.min_cp)
            range_class = PerformanceAnalyzer.get_cp_range_classification(airfoil_obj.cp_range)
            
            perf_data.append([
                f"Airfoil {len(perf_data)}",
                airfoil_obj.naca_code,
                f"{airfoil_obj.min_cp:.3f}",
                f"{airfoil_obj.max_cp:.3f}",
                f"{airfoil_obj.cp_range:.3f}",
                suction_class,
                range_class
            ])
        
        perf_table = Table(perf_data, colWidths=[0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1*inch, 1*inch])
        perf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), REPORT_CFG.TABLE_HEADER_COLOR),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(perf_table)
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_engineering_recommendations(self, airfoil_data_list: List[dict]):
        """Create engineering recommendations section"""
        story = []
        
        story.append(Paragraph("Engineering Recommendations", self.styles['CustomHeader']))
        
        # Analyze the set of airfoils for patterns and recommendations
        camber_values = [data['data'].camber for data in airfoil_data_list]
        thickness_values = [data['data'].thickness for data in airfoil_data_list]
        min_cps = [data['data'].min_cp for data in airfoil_data_list]
        
        recommendations_text = f"""
        Based on the comparative analysis of {len(airfoil_data_list)} airfoils, the following 
        engineering insights and recommendations are provided:<br/><br/>
        
        <b>Design Trade-offs Observed:</b><br/>
        • Camber range: {min(camber_values)}% to {max(camber_values)}%<br/>
        • Thickness range: {min(thickness_values)}% to {max(thickness_values)}%<br/>
        • Peak suction variation: {min(min_cps):.3f} to {max(min_cps):.3f}<br/><br/>
        
        <b>Application-Specific Insights:</b><br/>
        """
        
        # Add specific recommendations based on airfoil types present
        applications = set(data['config']['selected_preset'] for data in airfoil_data_list)
        
        if "Military Fighter Aircraft" in applications:
            recommendations_text += "• Symmetric airfoils (NACA 00XX) excel in combat maneuverability<br/>"
        
        if "Commercial Passenger Aircraft" in applications:
            recommendations_text += "• Moderate camber airfoils optimize cruise efficiency<br/>"
        
        if "Cargo Transport Aircraft" in applications:
            recommendations_text += "• High camber configurations maximize payload capability<br/>"
        
        recommendations_text += """<br/><b>Selection Guidelines:</b><br/>
        • Choose symmetric airfoils for bidirectional performance requirements<br/>
        • Select moderate camber (2-3%) for balanced cruise efficiency<br/>
        • Use high camber (4%+) for maximum lift applications<br/>
        • Consider thickness ratio based on structural requirements<br/>
        • Evaluate operational envelope compatibility
        """
        
        story.append(Paragraph(recommendations_text, self.styles['CustomBody']))
        
        return story
    
    def _create_comparison_appendix(self, airfoil_data_list: List[dict]):
        """Create detailed comparison appendix"""
        story = []
        
        story.append(PageBreak())
        story.append(Paragraph("Appendix: Detailed Comparison Data", self.styles['CustomHeader']))
        
        # Statistical summary
        story.append(Paragraph("Statistical Summary", self.styles['CustomBody']))
        
        comparison_stats = PerformanceAnalyzer.compare_airfoils([data['data'] for data in airfoil_data_list])
        
        stats_text = f"""
        <b>Descriptive Statistics:</b><br/>
        • Average minimum Cp: {comparison_stats['stats']['avg_min_cp']:.4f}<br/>
        • Standard deviation: {comparison_stats['stats']['std_min_cp']:.4f}<br/>
        • Average thickness: {comparison_stats['stats']['avg_thickness']:.1f}%<br/>
        • Average camber: {comparison_stats['stats']['avg_camber']:.1f}%<br/><br/>
        
        <b>Performance Leaders:</b><br/>
        • Best suction: {comparison_stats['naca_codes'][comparison_stats['stats']['best_suction_idx']]}<br/>
        • Highest Cp range: {comparison_stats['naca_codes'][comparison_stats['stats']['highest_cp_range_idx']]}
        """
        
        story.append(Paragraph(stats_text, self.styles['CustomBody']))
        
        return story

# ============================================================================
# PHASE 7: USER INTERFACE COMPONENTS
# ============================================================================

class UIComponents:
    """Reusable UI components for the application"""
    
    @staticmethod
    def create_app_header():
        """Create the main application header"""
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); border-radius: 10px; margin-bottom: 2rem;">
            <h1 style="color: white; margin: 0; font-size: 2.5rem;">✈️ Neural Network Airfoil Cp Predictor</h1>
            <p style="color: #e8f4f8; margin: 0.5rem 0 0 0; font-size: 1.2rem;">Rapid Airfoil Screening Platform</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_navigation_buttons():
        """Create navigation buttons for different analysis modes"""
        st.markdown("## ⚙️ Choose Analysis Mode")
        
        col_nav1, col_nav2, col_nav3 = st.columns(3)
        
        with col_nav1:
            single_mode = st.button(
                "🔍 Single Airfoil Analysis",
                use_container_width=True,
                help="Detailed analysis of one airfoil configuration",
                type="primary" if SessionStateManager.get('page_mode') == 'single' else "secondary"
            )
        
        with col_nav2:
            multiple_mode = st.button(
                "📊 Multiple Airfoil Comparison",
                use_container_width=True,
                help="Compare 2-4 airfoils side-by-side",
                type="primary" if SessionStateManager.get('page_mode') == 'multiple' else "secondary"
            )
        
        with col_nav3:
            quiz_mode = st.button(
                "💭 Airfoil Knowledge Quiz",
                use_container_width=True,
                help="Test your airfoil aerodynamics knowledge",
                type="primary" if SessionStateManager.get('page_mode') == 'quiz' else "secondary"
            )
        
        return single_mode, multiple_mode, quiz_mode
    
    @staticmethod
    def display_current_mode_info(current_mode):
        """Display information about the current analysis mode"""
        mode_descriptions = {
            'single': "🔍 **Single Airfoil Analysis** - Detailed examination of one airfoil configuration",
            'multiple': "📊 **Multiple Airfoil Comparison** - Side-by-side analysis of 2-4 airfoils",
            'quiz': "💭 **Knowledge Quiz** - Test and improve your understanding of airfoil aerodynamics"
        }
        
        st.info(mode_descriptions[current_mode])
        st.markdown("---")
    
    @staticmethod
    def create_metrics_display(airfoil_data: 'AirfoilData', selected_preset: str):
        """Create metrics display for airfoil data"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            aircraft_type = selected_preset.split()[0] if selected_preset != "Custom (Manual Input)" else "Custom"
            st.metric("🛫 Aircraft Type", aircraft_type)
        with col2:
            st.metric(" NACA Code", airfoil_data.naca_code)
        with col3:
            st.metric(" Camber", f"{airfoil_data.camber}%")
        with col4:
            position_display = f"{airfoil_data.position*10}%" if airfoil_data.camber > 0 else "N/A"
            st.metric(" Position", position_display)
        with col5:
            st.metric(" Thickness", f"{airfoil_data.thickness}%")
    
    @staticmethod
    def create_prediction_statistics(airfoil_data: 'AirfoilData'):
        """Create prediction statistics display"""
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        
        with col_stats1:
            st.metric("Min Cp (Peak Suction)", f"{airfoil_data.min_cp:.3f}")
        with col_stats2:
            st.metric("Max Cp (Max Pressure)", f"{airfoil_data.max_cp:.3f}")
        with col_stats3:
            st.metric("Cp Range", f"{airfoil_data.cp_range:.3f}")
        with col_stats4:
            st.metric("Upper Surface Min", f"{airfoil_data.cp_upper.min():.3f}")

class SidebarManager:
    """Manages sidebar configuration and parameters"""
    
    @staticmethod
    def create_sidebar_controls():
        """Create sidebar controls for single airfoil analysis"""
        st.sidebar.markdown("## Aircraft Configuration Selection")
        
        preset_names = AircraftPresets.get_preset_names()
        selected_preset = st.sidebar.selectbox(
            "**Choose Aircraft Type**",
            preset_names,
            index=0,
            help="Select a preset aircraft configuration"
        )
        
        if selected_preset != "Custom (Manual Input)":
            preset_info = AircraftPresets.get_preset_by_name(selected_preset)
            st.sidebar.info(f"**NACA {preset_info['naca']}**\n\n{preset_info['description']}")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("## NACA Parameters")
        
        return selected_preset
    
    @staticmethod
    def create_parameter_sliders(selected_preset):
        """Create parameter sliders based on selected preset"""
        if selected_preset == "Custom (Manual Input)":
            default_camber = 2
            default_position = 4
            default_thickness = 12
        else:
            preset_config = AircraftPresets.get_preset_by_name(selected_preset)
            default_camber = preset_config['camber']
            default_position = preset_config['position']
            default_thickness = preset_config['thickness']
        
        # Parameter sliders
        camber = st.sidebar.slider(
            "**Camber (M)**", 
            min_value=AIRFOIL_CFG.NACA_CAMBER_MIN,
            max_value=AIRFOIL_CFG.NACA_CAMBER_MAX,
            value=default_camber,
            step=1,
            help="Maximum camber as percentage of chord (0-7%)"
        )
        
        if camber > 0:
            position = st.sidebar.slider(
                "**Camber Position (P)**",
                min_value=AIRFOIL_CFG.NACA_POSITION_MIN,
                max_value=AIRFOIL_CFG.NACA_POSITION_MAX,
                value=default_position,
                step=1,
                help="Position of maximum camber (20-60% of chord)"
            )
        else:
            position = 4
            st.sidebar.markdown("*Camber position not applicable for symmetric airfoils*")
        
        thickness = st.sidebar.slider(
            "**Thickness (XX)**",
            min_value=AIRFOIL_CFG.NACA_THICKNESS_MIN,
            max_value=AIRFOIL_CFG.NACA_THICKNESS_MAX,
            value=default_thickness,
            step=1,
            help="Maximum thickness as percentage of chord (6-30%)"
        )
        
        # Generate and display NACA code
        naca_code = NACAValidator.generate_naca_code(camber, position, thickness)
        st.sidebar.markdown(f"### Current Configuration\n**NACA {naca_code}**")
        
        return camber, position, thickness, naca_code
    
    @staticmethod
    def create_model_info_sidebar(model_manager):
        """Create model information in sidebar"""
        if model_manager.is_loaded:
            st.sidebar.markdown("---")
            st.sidebar.markdown("## Model Status")
            st.sidebar.success("Model Loaded")
            
            model_info = model_manager.get_model_info()
            st.sidebar.markdown(f"**Predictions Made:** {model_info['predictions_made']}")
            st.sidebar.markdown(f"**Load Time:** {model_info['load_time']}")
            
            with st.sidebar.expander("Model Details"):
                st.markdown(f"**Input Shape:** {model_info['input_shape']}")
                st.markdown(f"**Output Shape:** {model_info['output_shape']}")
        else:
            st.sidebar.markdown("---")
            st.sidebar.error("Model Not Loaded")


class ExportManager:
    """Handles data export functionality"""
    
    @staticmethod
    def create_export_buttons(airfoil_data: 'AirfoilData', selected_preset: str, analysis_plots: list = None):
        """Create export buttons for single airfoil analysis"""
        st.markdown("#### Export Results")
        
               
        # Prepare data for export
        results_data = pd.DataFrame({
            'x_c': airfoil_data.x_cp,
            'Cp_upper': airfoil_data.cp_upper,
            'Cp_lower': airfoil_data.cp_lower
        })
        
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            ExportManager._create_csv_download(results_data, airfoil_data.naca_code)
        
        with col_export2:
            ExportManager._create_text_report_download(airfoil_data, selected_preset)
        
        with col_export3:
            ExportManager._create_pdf_download(airfoil_data, selected_preset, analysis_plots)
    
    @staticmethod
    def _create_csv_download(results_data: pd.DataFrame, naca_code: str):
        """Create CSV download button"""
        # Use pre-computed CSV data from session state
        csv_data = st.session_state.export_data['csv_data']
        st.download_button(
            label="Download Cp Data (CSV)",
            data=csv_data,
            file_name=f"NACA_{naca_code}_Cp_distribution.csv",
            mime="text/csv",
            help="Download pressure coefficient data in CSV format"
        )
    
    @staticmethod
    def _create_text_report_download(airfoil_data: 'AirfoilData', selected_preset: str):
        """Create text report download button"""
        # Use pre-computed text data from session state
        summary_text = st.session_state.export_data['txt_data']
    
        st.download_button(
            label="Download Analysis Report (TXT)",
            data=summary_text,
            file_name=f"NACA_{airfoil_data.naca_code}_analysis_report.txt",
            mime="text/plain",
            help="Download complete analysis report in text format"
        )
    @staticmethod
    def _create_pdf_download(airfoil_data: 'AirfoilData', selected_preset: str, analysis_plots: list = None):
        """Create PDF download button"""
        if REPORTLAB_AVAILABLE:
            if st.button("Generate & Download PDF Report", help="Generate and download comprehensive PDF report"):
                try:
                    with st.spinner("Generating PDF report..."):
                        pdf_generator = PDFReportGenerator()
                        pdf_buffer = pdf_generator.generate_single_airfoil_report(
                            airfoil_data, selected_preset, analysis_plots or []
                        )
                        
                        # Force immediate download
                        st.download_button(
                            label="Click Here to Download PDF",
                            data=pdf_buffer.getvalue(),
                            file_name=f"NACA_{airfoil_data.naca_code}_comprehensive_report.pdf",
                            mime="application/pdf",
                            help="Download comprehensive PDF analysis report",
                            key=f"pdf_download_{airfoil_data.naca_code}"
                        )
                        
                        st.success("PDF report generated! Click the download button above.")
                        
                except Exception as e:
                    st.error(f"PDF generation failed: {str(e)}")
        else:
            st.info("PDF reports require 'pip install reportlab'")
    
   
    

class AnalysisController:
    """Controls the single airfoil analysis workflow"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
    
    def run_single_analysis(self):
        """Main single airfoil analysis workflow"""
        # Create sidebar controls
        selected_preset = SidebarManager.create_sidebar_controls()
        
        # Create parameter sliders
        camber, position, thickness, naca_code = SidebarManager.create_parameter_sliders(selected_preset)
        
        # Add model info to sidebar
        SidebarManager.create_model_info_sidebar(self.model_manager)
        
        # Create airfoil data object
        airfoil_data = AirfoilData(naca_code, camber, position, thickness)
        
        # Display metrics
        UIComponents.create_metrics_display(airfoil_data, selected_preset)
        
        # Show preset description
        if selected_preset != "Custom (Manual Input)":
            preset_info = AircraftPresets.get_preset_by_name(selected_preset)
            st.info(f"**{selected_preset}**: {preset_info['description']}")
        
        st.markdown("---")
        
        # Generate and display airfoil preview
        self._generate_and_display_preview(airfoil_data, selected_preset)
        
        # Technical details section
        if selected_preset != "Custom (Manual Input)":
            self._display_technical_details(selected_preset)
        
        # Prediction section
        self._handle_prediction_workflow(airfoil_data, selected_preset)
    
    def _generate_and_display_preview(self, airfoil_data: 'AirfoilData', selected_preset: str):
        """Generate coordinates and display airfoil preview"""
        try:
            with st.spinner("Generating airfoil coordinates..."):
                x_coords, y_coords = CoordinateGenerator.generate_naca_coordinates(airfoil_data.naca_code)
                airfoil_data.x_coords = x_coords
                airfoil_data.y_coords = y_coords
            
            st.markdown("### Airfoil Geometry Preview")
            fig_preview = AirfoilVisualizer.create_airfoil_preview(airfoil_data, selected_preset)
            st.pyplot(fig_preview)
            PlotUtilities.cleanup_figures()
            
        except Exception as e:
            ErrorHandler.handle_coordinate_generation_error(str(e))
    
    def _display_technical_details(self, selected_preset: str):
        """Display technical details for preset configurations"""
        preset_info = AircraftPresets.get_preset_by_name(selected_preset)
        
        with st.expander("Technical Details"):
            st.markdown("### Design Philosophy")
            st.write(preset_info['technical_details']['design_philosophy'])
            
            st.markdown("### Performance Characteristics")
            st.write(preset_info['technical_details']['performance_characteristics'])
            
            st.markdown("### Structural Considerations")
            st.write(preset_info['technical_details']['structural_considerations'])
            
            st.markdown("### Operational Envelope")
            st.write(preset_info['technical_details']['operational_envelope'])

        with st.expander("Application Context"):
            context = preset_info['application_context']
            
            col_ctx1, col_ctx2 = st.columns(2)
            
            with col_ctx1:
                st.markdown("### Primary Applications")
                st.write(f"**Primary Use:** {context['primary_use']}")
                st.write(f"**Aircraft Examples:** {context['aircraft_examples']}")
            
            with col_ctx2:
                st.markdown("### Design Rationale")
                st.write(context['design_rationale'])
            
            st.markdown("### Performance Notes")
            st.info(context['performance_notes'])
    
    def _handle_prediction_workflow(self, airfoil_data: 'AirfoilData', selected_preset: str):
        """Handle the prediction workflow"""
        st.markdown("### Neural Network Prediction")
        
        col_pred1, col_pred2 = st.columns([1, 2])
        
        with col_pred1:
            st.markdown("**Ready to predict Cp distribution?**")
            predict_button = st.button("Predict Cp Distribution", type="primary", use_container_width=True)
        
        with col_pred2:
            self._display_model_information()
        
        if predict_button:
            self._execute_prediction(airfoil_data, selected_preset)

        # Display results if prediction is complete
        if st.session_state.get('prediction_complete', False):
            airfoil_data = st.session_state.airfoil_data
            selected_preset = st.session_state.selected_preset
            
            ErrorHandler.display_success_message("Aerodynamic analysis completed successfully!")
            
            # Display statistics
            UIComponents.create_prediction_statistics(airfoil_data)
            
            # Create and display comprehensive results
            self._display_comprehensive_results(airfoil_data, selected_preset)
            
            # Display performance interpretation
            if selected_preset != "Custom (Manual Input)":
                self._display_performance_interpretation(airfoil_data, selected_preset)
            
            # Export options
            if 'analysis_plots' not in st.session_state:
                st.session_state.analysis_plots = [AirfoilVisualizer.create_cp_analysis_plots(airfoil_data)]
            ExportManager.create_export_buttons(airfoil_data, selected_preset, st.session_state.analysis_plots)
            
            # Data table
            self._display_data_table(airfoil_data)
    
    def _display_model_information(self):
        """Display neural network model information"""
        with st.expander("Neural Network Model Information", expanded=False):
            st.markdown("### Model Specifications")
            
            col_model1, col_model2 = st.columns(2)
            
            with col_model1:
                st.markdown("""
                **Architecture:**
                - Deep feedforward neural network
                - Multiple hidden layers with ReLU activation
                - Trained on 1000+ NACA airfoil configurations
                
                **Input Processing:**
                - 400 scaled airfoil coordinates (200 x,y pairs)
                - Normalized to [0,1] range for optimal training
                - Arc-length parameterization for consistency
                """)
            
            with col_model2:
                st.markdown("""
                **Output Generation:**
                - 200 pressure coefficient values
                - 100 upper surface + 100 lower surface points
                - Inviscid flow assumptions
                
                **Operating Conditions:**
                - Zero angle of attack
                - Subsonic flow (M < 0.8)
                - Standard atmospheric conditions
                """)
            
            st.markdown("### Performance Characteristics")
            st.success("""**Prediction Time:** < 1 second  
            **Accuracy:** 95%+ vs CFD  
            **Validation:** Cross-validated on 300+ test cases""")
            
            st.markdown("### Technical Capabilities")
            st.info("""
            **Advantages:** Instant prediction vs. hours of CFD | Consistent results | No mesh generation required
            
            **Limitations:** Zero angle of attack only | Inviscid assumptions | NACA 4-digit airfoils only
            """)
    
    def _execute_prediction(self, airfoil_data: 'AirfoilData', selected_preset: str):
        """Execute the neural network prediction"""
        st.markdown("---")
        st.markdown("### Prediction Results")
        
        try:
            with st.spinner("Analyzing airfoil aerodynamics..."):
                # Scale coordinates for prediction
                scaled_input = CoordinateProcessor.scale_coordinates_for_prediction(
                    airfoil_data.x_coords, airfoil_data.y_coords
                )
                
                # Make prediction
                prediction = self.model_manager.predict(scaled_input)
                cp_distribution = prediction[0]
                
                # Process results
                airfoil_data.cp_distribution = cp_distribution
                airfoil_data.cp_upper = cp_distribution[:AIRFOIL_CFG.CP_UPPER_POINTS]
                airfoil_data.cp_lower = cp_distribution[AIRFOIL_CFG.CP_LOWER_POINTS:]
                airfoil_data.x_cp = np.linspace(0, 1, AIRFOIL_CFG.CP_UPPER_POINTS)
                airfoil_data.prediction_successful = True
                # Store prediction results in session state
                st.session_state.prediction_complete = True
                st.session_state.airfoil_data = airfoil_data
                st.session_state.selected_preset = selected_preset
                # Pre-generate export data for faster downloads
                results_data = pd.DataFrame({
                'x_c': airfoil_data.x_cp,
                'Cp_upper': airfoil_data.cp_upper,
                'Cp_lower': airfoil_data.cp_lower
                })

                summary_text = f"""NACA {airfoil_data.naca_code} Analysis Report
                Generated by Neural Network Model

                Aircraft Configuration: {selected_preset}

                Airfoil Parameters:
                - Camber: {airfoil_data.camber}%
                - Camber Position: {airfoil_data.position*10}%  
                - Thickness: {airfoil_data.thickness}%

                Pressure Coefficient Results:
                - Minimum Cp: {airfoil_data.min_cp:.4f}
                - Maximum Cp: {airfoil_data.max_cp:.4f}
                - Cp Range: {airfoil_data.cp_range:.4f}
                - Upper Surface Min Cp: {airfoil_data.cp_upper.min():.4f}
                - Lower Surface Max Cp: {airfoil_data.cp_lower.max():.4f}

                Analysis Timestamp: {airfoil_data.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}
                """

                # Store pre-computed export data
                st.session_state.export_data = {
                'csv_data': results_data.to_csv(index=False),
                'txt_data': summary_text,
                'results_dataframe': results_data,
                'naca_code': airfoil_data.naca_code,
                'selected_preset': selected_preset
                }
            
            
            # Data table
            self._display_data_table(airfoil_data)
            
        except Exception as e:
            ErrorHandler.handle_prediction_error(str(e))
    
    def _display_comprehensive_results(self, airfoil_data: 'AirfoilData', selected_preset: str):
        """Display comprehensive analysis results"""
        st.markdown("#### Airfoil Analysis Results")
        
        fig_results = AirfoilVisualizer.create_cp_analysis_plots(airfoil_data)
        st.pyplot(fig_results)
        PlotUtilities.cleanup_figures()
    
    def _display_performance_interpretation(self, airfoil_data: 'AirfoilData', selected_preset: str):
        """Display engineering interpretation and performance assessment"""
        st.markdown("---")
        with st.expander("Engineering Interpretation & Application Context", expanded=False):
            col_interp1, col_interp2 = st.columns(2)
        
        with col_interp1:
            st.markdown("### Performance Assessment")
            
            # Get performance rating
            performance_rating, performance_color, performance_note = PerformanceAnalyzer.get_performance_rating(
                airfoil_data, selected_preset
            )
            
            if performance_color == "success":
                st.success(f"**Performance Rating:** {performance_rating}")
            elif performance_color == "warning":
                st.warning(f"**Performance Rating:** {performance_rating}")
            else:
                st.info(f"**Performance Rating:** {performance_rating}")
            
            st.write(performance_note)
            
            # Key metrics interpretation
            st.markdown("**Key Metrics for this Application:**")
            suction_class = PerformanceAnalyzer.get_suction_classification(airfoil_data.min_cp)
            range_class = PerformanceAnalyzer.get_cp_range_classification(airfoil_data.cp_range)
            
            st.write(f"• **Peak Suction:** {airfoil_data.min_cp:.3f} - {suction_class} suction")
            st.write(f"• **Pressure Range:** {airfoil_data.cp_range:.3f} - {range_class} lift potential")
        
        with col_interp2:
            st.markdown("### Real-World Implications")
            
            # Application-specific insights
            self._display_application_insights(airfoil_data, selected_preset)
        
        # Overall assessment
            self._display_overall_assessment(airfoil_data, selected_preset)
    
    def _display_application_insights(self, airfoil_data: 'AirfoilData', selected_preset: str):
        """Display application-specific insights"""
        st.markdown("**For this aircraft type:**")
        
        min_cp_abs = abs(airfoil_data.min_cp)
        
        if selected_preset == "Commercial Passenger Aircraft":
            fuel_efficiency = "Excellent" if 1.0 < min_cp_abs < 2.5 else "Good"
            st.write(f"""
            • **Fuel Efficiency:** {fuel_efficiency}  
            • **Passenger Comfort:** Stable pressure distribution reduces turbulence  
            • **Operating Economics:** Balanced performance for airline operations  
            • **Safety Margin:** Predictable stall characteristics important for passenger safety
            """)
        
        elif selected_preset == "Military Fighter Aircraft":
            combat_performance = "Excellent" if min_cp_abs > 6.0 else "Good"
            st.write(f"""
            • **Combat Performance:** {combat_performance} maneuverability  
            • **Speed Capability:** Symmetric design enables inverted flight  
            • **Structural Loads:** Thin section handles high-g maneuvers  
            • **Mission Flexibility:** Identical performance upright and inverted
            """)
        
        elif selected_preset == "Cargo Transport Aircraft":
            payload_capacity = "Maximum" if min_cp_abs > 4.0 else "High"
            st.write(f"""
            • **Payload Capacity:** {payload_capacity} lift for heavy loads  
            • **Short Field Performance:** High camber enables short runway operations  
            • **Structural Capability:** Thick section provides cargo volume
            • **Operational Flexibility:** Designed for varying load conditions  
            """)
        
        else:
            st.write(f"""
            • **Operational Efficiency:** Designed for specific mission requirements  
            • **Performance Balance:** Optimized for intended flight envelope  
            • **Handling Qualities:** Appropriate for pilot skill level  
            • **Mission Capability:** Meets design objectives effectively
            """)
    
    def _display_overall_assessment(self, airfoil_data: 'AirfoilData', selected_preset: str):
        """Display overall performance assessment"""
        st.markdown("### Overall Assessment")
        
        context = AircraftPresets.get_preset_by_name(selected_preset)['application_context']
        performance_rating, performance_color, _ = PerformanceAnalyzer.get_performance_rating(
            airfoil_data, selected_preset
        )
        
        assessment_text = f"""
        The NACA {airfoil_data.naca_code} airfoil demonstrates characteristics well-suited for **{context['primary_use']}**. 
        
        **Design Validation:** {context['design_rationale']}
        
        **Predicted Performance:** The pressure distribution shows {performance_rating.lower()} characteristics for this application, 
        with peak suction of {airfoil_data.min_cp:.3f} and pressure range of {airfoil_data.cp_range:.3f}.
        """
        
        if performance_color == "success":
            st.success(assessment_text)
        elif performance_color == "warning":
            st.warning(assessment_text)
        else:
            st.info(assessment_text)
    
    def _display_data_table(self, airfoil_data: 'AirfoilData'):
        """Display detailed data table"""
        with st.expander("View Detailed Cp Data"):
            results_data = pd.DataFrame({
                'x_c': airfoil_data.x_cp,
                'Cp_upper': airfoil_data.cp_upper,
                'Cp_lower': airfoil_data.cp_lower
            })
            
            st.dataframe(results_data, use_container_width=True)
            
            st.markdown("**Quick Statistics:**")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.write(f"Upper Surface Mean Cp: {airfoil_data.cp_upper.mean():.3f}")
            with col_stat2:
                st.write(f"Lower Surface Mean Cp: {airfoil_data.cp_lower.mean():.3f}")
            with col_stat3:
                st.write(f"Overall Mean Cp: {airfoil_data.cp_distribution.mean():.3f}")

# ============================================================================
# PAGE MODE MANAGEMENT
# ============================================================================

class PageModeManager:
    """Manages page mode state and navigation"""
    
    @staticmethod
    def initialize_session_state():
        """Initialize session state variables"""
        SessionStateManager.initialize()
    
    @staticmethod
    def handle_navigation(single_mode, multiple_mode, quiz_mode):
        """Handle navigation between different modes"""
        if single_mode:
            SessionStateManager.set('page_mode', 'single')
            st.rerun()
        elif multiple_mode:
            SessionStateManager.set('page_mode', 'multiple')
            st.rerun()
        elif quiz_mode:
            SessionStateManager.set('page_mode', 'quiz')
            st.rerun()
        
        return SessionStateManager.get('page_mode', 'single')



# ============================================================================
# PHASE 9: MULTIPLE AIRFOIL COMPARISON SYSTEM
# ============================================================================

class MultipleAirfoilAnalyzer:
    """Handles multiple airfoil comparative analysis workflow"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
    
    def run_analysis(self):
        """Main multiple airfoil analysis workflow"""
        st.markdown("## Multiple Airfoil Comparative Analysis")
        st.markdown("Compare up to 4 different airfoils side-by-side to understand their aerodynamic characteristics.")
        st.markdown("---")
        
        # Configuration section
        num_airfoils, quick_preset = self._create_configuration_controls()
        
        # Create airfoil configuration interface
        airfoil_configs = self._create_airfoil_configurations(num_airfoils, quick_preset)
        
        # Display technical comparison overview
        self._display_technical_overview(airfoil_configs)
        
        # Analysis execution
        if self._create_analysis_button():
            self._execute_comparative_analysis(airfoil_configs)
        # Display results if comparison is complete
        if st.session_state.get('comparison_complete', False):
          airfoil_data = st.session_state.comparison_data
          self._display_comparison_results(airfoil_data)
    
    def _create_configuration_controls(self):
        """Create configuration controls for multiple airfoil analysis"""
        st.sidebar.markdown("## Analysis Configuration")
        num_airfoils = st.sidebar.selectbox(
            "**Number of Airfoils to Compare**",
            [2, 3, 4],
            index=0,
            help="Select how many airfoils you want to analyze simultaneously"
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("## Quick Presets")
        
        preset_comparisons = PresetComparisons.get_comparison_presets()
        preset_names = ["Custom Configuration"] + list(preset_comparisons.keys())
        
        quick_preset = st.sidebar.selectbox(
            "**Quick Comparison Presets**",
            preset_names,
            index=0,
            help="Select a predefined comparison or choose Custom for manual setup"
        )
        
        if quick_preset != "Custom Configuration":
            preset_info = preset_comparisons[quick_preset]
            st.sidebar.info(f"**{quick_preset}**\n\n{preset_info['description']}\n\nFocus: {preset_info['analysis_focus']}")
        
        return num_airfoils, quick_preset
    
    def _create_airfoil_configurations(self, num_airfoils, quick_preset):
        """Create configuration interface for each airfoil"""
        # Create column layout based on number of airfoils
        if num_airfoils == 2:
            containers = st.columns(2)
        elif num_airfoils == 3:
            containers = st.columns(3)
        else:  # 4 airfoils
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)
            containers = [col1, col2, col3, col4]
        
        # Get preset configuration if applicable
        preset_config = None
        if quick_preset != "Custom Configuration":
            preset_comparisons = PresetComparisons.get_comparison_presets()
            preset_config = preset_comparisons[quick_preset]['airfoils']
        
        airfoil_configs = {}
        
        for i in range(num_airfoils):
            # Determine default preset
            if preset_config and i < len(preset_config):
                default_preset = preset_config[i]
            else:
                default_preset = "Custom (Manual Input)"
            
            # Create configuration for this airfoil
            config = self._configure_single_airfoil(containers[i], i+1, default_preset)
            airfoil_configs[f"airfoil_{i+1}"] = config
        
        return airfoil_configs
    
    def _configure_single_airfoil(self, container, airfoil_num, default_preset):
        """Configure a single airfoil in the given container"""
        with container:
            st.markdown(f"### Airfoil {airfoil_num}")
            
            # Aircraft type selector
            preset_names = AircraftPresets.get_preset_names()
            selected_preset = st.selectbox(
                "**Aircraft Type**",
                preset_names,
                index=preset_names.index(default_preset) if default_preset in preset_names else 0,
                key=f"preset_airfoil_{airfoil_num}",
                help="Select a preset aircraft configuration or choose 'Custom' for manual input"
            )
            
            # Show preset information
            if selected_preset != "Custom (Manual Input)":
                preset_info = AircraftPresets.get_preset_by_name(selected_preset)
                st.info(f"**NACA {preset_info['naca']}**\n{preset_info['description']}")
            
            # Parameter configuration
            if selected_preset == "Custom (Manual Input)":
                default_camber, default_position, default_thickness = 2, 4, 12
            else:
                preset_config = AircraftPresets.get_preset_by_name(selected_preset)
                default_camber = preset_config['camber']
                default_position = preset_config['position']
                default_thickness = preset_config['thickness']
            
            # Parameter sliders
            camber = st.slider(
                "**Camber (M)**",
                min_value=AIRFOIL_CFG.NACA_CAMBER_MIN,
                max_value=AIRFOIL_CFG.NACA_CAMBER_MAX,
                value=default_camber,
                step=1,
                key=f"camber_airfoil_{airfoil_num}",
                help="Maximum camber as percentage of chord (0-7%)"
            )
            
            if camber > 0:
                position = st.slider(
                    "**Camber Position (P)**",
                    min_value=AIRFOIL_CFG.NACA_POSITION_MIN,
                    max_value=AIRFOIL_CFG.NACA_POSITION_MAX,
                    value=default_position,
                    step=1,
                    key=f"position_airfoil_{airfoil_num}",
                    help="Position of maximum camber (20-60% of chord)"
                )
            else:
                position = 4
                st.markdown("*Position not applicable for symmetric airfoils*")
            
            thickness = st.slider(
                "**Thickness (XX)**",
                min_value=AIRFOIL_CFG.NACA_THICKNESS_MIN,
                max_value=AIRFOIL_CFG.NACA_THICKNESS_MAX,
                value=default_thickness,
                step=1,
                key=f"thickness_airfoil_{airfoil_num}",
                help="Maximum thickness as percentage of chord (6-30%)"
            )
            
            # Generate NACA code
            naca_code = NACAValidator.generate_naca_code(camber, position, thickness)
            st.markdown(f"**Current: NACA {naca_code}**")
            
            return {
                'naca_code': naca_code,
                'selected_preset': selected_preset,
                'camber': camber,
                'position': position,
                'thickness': thickness,
                'color': VIZ_CFG.AIRFOIL_COLORS[airfoil_num-1],
                'linestyle': VIZ_CFG.AIRFOIL_LINE_STYLES[airfoil_num-1]
            }
    
    def _display_technical_overview(self, airfoil_configs):
        """Display technical comparison overview"""
        st.markdown("### Technical Comparison Overview")
        
        with st.expander("Configured Airfoils Summary", expanded=True):
            comparison_summary = []
            
            for i, (airfoil_key, config) in enumerate(airfoil_configs.items()):
                if config['selected_preset'] != "Custom (Manual Input)":
                    preset_info = AircraftPresets.get_preset_by_name(config['selected_preset'])
                    primary_use = preset_info['application_context']['primary_use']
                else:
                    primary_use = 'User-defined'
                
                summary_row = {
                    'Airfoil': f"Airfoil {i+1}",
                    'NACA': config['naca_code'],
                    'Aircraft Type': config['selected_preset'],
                    'Primary Use': primary_use,
                    'Camber (%)': config['camber'],
                    'Thickness (%)': config['thickness']
                }
                comparison_summary.append(summary_row)
            
            summary_df = pd.DataFrame(comparison_summary)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    def _create_analysis_button(self):
        """Create the analysis execution button"""
        col_analyze1, col_analyze2, col_analyze3 = st.columns([1, 2, 1])
        
        with col_analyze2:
            return st.button(
                "Analyze All Airfoils",
                type="primary",
                use_container_width=True,
                help="Generate coordinates and predict Cp distributions for all configured airfoils"
            )
    
    def _execute_comparative_analysis(self, airfoil_configs):
        """Execute the comparative analysis for all airfoils"""
        st.markdown("---")
        st.markdown("## Comparative Analysis Results")
        
        # Generate coordinates and predictions for all airfoils
        airfoil_data = {}
        
        with st.spinner("Generating coordinates and making predictions for all airfoils..."):
            for airfoil_key, config in airfoil_configs.items():
                try:
                    # Create airfoil data object
                    airfoil_data_obj = AirfoilData(
                        config['naca_code'], 
                        config['camber'], 
                        config['position'], 
                        config['thickness']
                    )
                    
                    # Generate coordinates
                    x_coords, y_coords = CoordinateGenerator.generate_naca_coordinates(config['naca_code'])
                    airfoil_data_obj.x_coords = x_coords
                    airfoil_data_obj.y_coords = y_coords
                    
                    # Scale coordinates and make prediction
                    scaled_input = CoordinateProcessor.scale_coordinates_for_prediction(x_coords, y_coords)
                    prediction = self.model_manager.predict(scaled_input)
                    cp_distribution = prediction[0]
                    
                    # Process results
                    airfoil_data_obj.cp_distribution = cp_distribution
                    airfoil_data_obj.cp_upper = cp_distribution[:AIRFOIL_CFG.CP_UPPER_POINTS]
                    airfoil_data_obj.cp_lower = cp_distribution[AIRFOIL_CFG.CP_LOWER_POINTS:]
                    airfoil_data_obj.x_cp = np.linspace(0, 1, AIRFOIL_CFG.CP_UPPER_POINTS)
                    airfoil_data_obj.prediction_successful = True
                    
                    # Store with visualization properties
                    airfoil_data[airfoil_key] = {
                        'data': airfoil_data_obj,
                        'config': config
                    }
                    
                except Exception as e:
                    st.error(f"Error analyzing {config['naca_code']}: {str(e)}")
                    return
        
        ErrorHandler.display_success_message("All airfoils analyzed successfully!")

        # Store comparison results in session state
        st.session_state.comparison_complete = True
        st.session_state.comparison_data = airfoil_data
        
        
    
    def _display_comparison_results(self, airfoil_data):
        """Display comprehensive comparison results"""
        # Summary comparison table
        self._create_summary_table(airfoil_data)
        
        # Visualization section
        self._create_comparison_visualizations(airfoil_data)
        
        # Performance comparison charts
        self._create_performance_charts(airfoil_data)
        
        # Detailed metrics
        self._create_detailed_metrics(airfoil_data)
        
        # Performance insights
        self._create_performance_insights(airfoil_data)
        
        # Export options
        self._create_export_options(airfoil_data)
    
    def _create_summary_table(self, airfoil_data):
        """Create quick comparison summary table"""
        st.markdown("### Quick Comparison Summary")
        
        summary_data = []
        for airfoil_key, data in airfoil_data.items():
            airfoil_obj = data['data']
            config = data['config']
            aircraft_type = config['selected_preset'].split()[0] if config['selected_preset'] != "Custom (Manual Input)" else "Custom"
            
            summary_data.append({
                'Airfoil': f"Airfoil {airfoil_key.split('_')[1]}",
                'NACA Code': airfoil_obj.naca_code,
                'Aircraft Type': aircraft_type,
                'Camber (%)': airfoil_obj.camber,
                'Thickness (%)': airfoil_obj.thickness,
                'Min Cp': f"{airfoil_obj.min_cp:.3f}",
                'Max Cp': f"{airfoil_obj.max_cp:.3f}",
                'Cp Range': f"{airfoil_obj.cp_range:.3f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    def _create_comparison_visualizations(self, airfoil_data):
        """Create comparative visualizations"""
        st.markdown("### Comparative Visualizations")
        
        # Convert to list format for ComparisonVisualizer
        airfoil_data_list = list(airfoil_data.values())
        
        fig_comparison = ComparisonVisualizer.create_comparison_plots(airfoil_data_list)
        st.pyplot(fig_comparison)
        PlotUtilities.cleanup_figures()
    
    def _create_performance_charts(self, airfoil_data):
        """Create performance comparison charts"""
        st.markdown("### Performance Comparison Charts")
        
        airfoil_data_list = list(airfoil_data.values())
        fig_performance = PerformanceVisualizer.create_performance_comparison_chart(airfoil_data_list)
        st.pyplot(fig_performance)
        PlotUtilities.cleanup_figures()
    
    def _create_detailed_metrics(self, airfoil_data):
        """Create detailed performance metrics comparison"""
        st.markdown("### Detailed Performance Comparison")
        
        metrics_data = []
        for airfoil_key, data in airfoil_data.items():
            airfoil_obj = data['data']
            config = data['config']
            aircraft_type = config['selected_preset'].split()[0] if config['selected_preset'] != "Custom (Manual Input)" else "Custom"
            
            metrics_data.append({
                'Airfoil': f"NACA {airfoil_obj.naca_code}",
                'Aircraft Type': aircraft_type,
                'Peak Suction (Min Cp)': airfoil_obj.min_cp,
                'Max Pressure (Max Cp)': airfoil_obj.max_cp,
                'Pressure Range': airfoil_obj.cp_range,
                'Upper Surface Min Cp': airfoil_obj.cp_upper.min(),
                'Lower Surface Max Cp': airfoil_obj.cp_lower.max(),
                'Upper Surface Mean Cp': airfoil_obj.cp_upper.mean(),
                'Lower Surface Mean Cp': airfoil_obj.cp_lower.mean(),
                'Overall Mean Cp': airfoil_obj.cp_distribution.mean()
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Display with color coding
        st.dataframe(
            metrics_df.style.format({
                'Peak Suction (Min Cp)': '{:.4f}',
                'Max Pressure (Max Cp)': '{:.4f}',
                'Pressure Range': '{:.4f}',
                'Upper Surface Min Cp': '{:.4f}',
                'Lower Surface Max Cp': '{:.4f}',
                'Upper Surface Mean Cp': '{:.4f}',
                'Lower Surface Mean Cp': '{:.4f}',
                'Overall Mean Cp': '{:.4f}'
            }).background_gradient(subset=['Peak Suction (Min Cp)', 'Pressure Range'], cmap='RdYlBu'),
            use_container_width=True,
            hide_index=True
        )
    
    def _create_performance_insights(self, airfoil_data):
     """Create performance insights and analysis"""
     with st.expander("Engineering Analysis & Application Comparison", expanded=False):
        col_analysis1, col_analysis2 = st.columns(2)
        
        with col_analysis1:
            st.markdown("#### Performance Rankings")
            
            # Extract performance metrics
            airfoils_for_ranking = []
            for data in airfoil_data.values():
                airfoils_for_ranking.append({
                    'naca': data['data'].naca_code,
                    'min_cp': data['data'].min_cp,
                    'cp_range': data['data'].cp_range,
                    'application': data['config']['selected_preset']
                })
            
            # Sort by peak suction
            airfoils_by_suction = sorted(airfoils_for_ranking, key=lambda x: x['min_cp'])
            st.markdown("**Peak Suction (Highest to Lowest):**")
            for i, airfoil in enumerate(airfoils_by_suction):
                st.write(f"{i+1}. NACA {airfoil['naca']}: {airfoil['min_cp']:.3f}")
            
            # Sort by pressure range
            airfoils_by_range = sorted(airfoils_for_ranking, key=lambda x: x['cp_range'], reverse=True)
            st.markdown("**Pressure Range (Highest to Lowest):**")
            for i, airfoil in enumerate(airfoils_by_range):
                st.write(f"{i+1}. NACA {airfoil['naca']}: {airfoil['cp_range']:.3f}")
        
        with col_analysis2:
            st.markdown("#### Design Trade-offs Observed")
            
            # Analyze design patterns
            camber_values = [data['data'].camber for data in airfoil_data.values()]
            thickness_values = [data['data'].thickness for data in airfoil_data.values()]
            min_cps = [data['data'].min_cp for data in airfoil_data.values()]
            
            avg_camber = np.mean(camber_values)
            avg_thickness = np.mean(thickness_values)
            
            st.write(f"**Average Camber:** {avg_camber:.1f}%")
            st.write(f"**Average Thickness:** {avg_thickness:.1f}%")
            st.write(f"**Peak Suction Range:** {min(min_cps):.3f} to {max(min_cps):.3f}")
            
            # Design insights
            if max(camber_values) - min(camber_values) > 2:
                st.info("**Camber Variation:** Significant camber differences suggest different design philosophies for lift generation")
            
            if max(thickness_values) - min(thickness_values) > 5:
                st.info("**Thickness Variation:** Large thickness differences indicate trade-offs between structure and aerodynamics")
    
    def _create_export_options(self, airfoil_data):
        """Create export options for comparison data"""
        st.markdown("### Export Comparison Data")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            # Export summary data
            summary_data = []
            for airfoil_key, data in airfoil_data.items():
                airfoil_obj = data['data']
                config = data['config']
                summary_data.append({
                    'Airfoil': f"Airfoil {airfoil_key.split('_')[1]}",
                    'NACA Code': airfoil_obj.naca_code,
                    'Aircraft Type': config['selected_preset'],
                    'Camber (%)': airfoil_obj.camber,
                    'Thickness (%)': airfoil_obj.thickness,
                    'Min Cp': airfoil_obj.min_cp,
                    'Max Cp': airfoil_obj.max_cp,
                    'Cp Range': airfoil_obj.cp_range
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_csv = summary_df.to_csv(index=False)
            
            st.download_button(
                label="Download Summary (CSV)",
                data=summary_csv,
                file_name=f"airfoil_comparison_summary_{len(airfoil_data)}airfoils.csv",
                mime="text/csv"
            )
        
        with col_exp2:
            # Create detailed Cp data export
            detailed_data = []
            for airfoil_key, data in airfoil_data.items():
                airfoil_obj = data['data']
                for i, x in enumerate(airfoil_obj.x_cp):
                    detailed_data.append({
                        'Airfoil': airfoil_obj.naca_code,
                        'x_c': x,
                        'Cp_upper': airfoil_obj.cp_upper[i],
                        'Cp_lower': airfoil_obj.cp_lower[i]
                    })
            
            detailed_df = pd.DataFrame(detailed_data)
            detailed_csv = detailed_df.to_csv(index=False)
            
            st.download_button(
                label="Download Detailed Cp Data (CSV)",
                data=detailed_csv,
                file_name=f"airfoil_detailed_cp_{len(airfoil_data)}airfoils.csv",
                mime="text/csv"
            )
        
        with col_exp3:
            # PDF comparison report
            if REPORTLAB_AVAILABLE:
                if st.button("Generate PDF Comparison Report", help="Generate comprehensive PDF comparison report"):
                    with st.spinner("Generating comparison PDF report..."):
                        try:
                            pdf_generator = PDFReportGenerator()
                            airfoil_data_list = list(airfoil_data.values())
                            
                            # Create comparison plots
                            comparison_plots = [
                                ComparisonVisualizer.create_comparison_plots(airfoil_data_list),
                                PerformanceVisualizer.create_performance_comparison_chart(airfoil_data_list)
                            ]
                            
                            pdf_buffer = pdf_generator.generate_comparison_report(
                                airfoil_data_list, comparison_plots
                            )
                            
                            st.download_button(
                                label="Download PDF Comparison Report",
                                data=pdf_buffer.getvalue(),
                                file_name=f"airfoil_comparison_report_{len(airfoil_data)}airfoils.pdf",
                                mime="application/pdf",
                                help="Download comprehensive PDF comparison report"
                            )
                            
                            st.success("PDF comparison report generated successfully!")
                            
                            # Clean up plots
                            for plot in comparison_plots:
                                plt.close(plot)
                                
                        except Exception as e:
                            st.error(f"PDF generation failed: {str(e)}")
            else:
                st.info("PDF reports require 'pip install reportlab'")
        
        # Final summary
        st.markdown("---")
        st.markdown("## Summary & Key Findings")
        
        # Performance comparison summary
        comparison_stats = PerformanceAnalyzer.compare_airfoils([data['data'] for data in airfoil_data.values()])
        
        summary_text = f"""
        **Comparison Analysis Summary:**
        
        **Airfoils Analyzed:** {len(airfoil_data)} configurations
        
        **Peak Suction Range:** {comparison_stats['stats']['avg_min_cp']:.3f} ± {comparison_stats['stats']['std_min_cp']:.3f}
        
        **Average Thickness:** {comparison_stats['stats']['avg_thickness']:.1f}%
        
        **Average Camber:** {comparison_stats['stats']['avg_camber']:.1f}%
        
        **Performance Leaders:**
        - **Best Suction:** {comparison_stats['naca_codes'][comparison_stats['stats']['best_suction_idx']]}
        - **Highest Cp Range:** {comparison_stats['naca_codes'][comparison_stats['stats']['highest_cp_range_idx']]}
        """
        
        st.info(summary_text)
        
        ErrorHandler.display_success_message("Comparative analysis completed successfully!")

# ============================================================================
# PHASE 10: AIRFOIL KNOWLEDGE QUIZ SYSTEM
# ============================================================================

class QuizQuestion:
    """Data class for individual quiz questions"""
    
    def __init__(self, question: str, options: list, correct_index: int, explanation: str, difficulty: str = "Beginner"):
        self.question = question
        self.options = options
        self.correct_index = correct_index
        self.explanation = explanation
        self.difficulty = difficulty
        
    @property
    def correct_answer(self):
        return self.options[self.correct_index]
    
    def is_correct(self, selected_index: int) -> bool:
        return selected_index == self.correct_index

class QuizDatabase:
    """Database of quiz questions organized by difficulty level"""
    
    @staticmethod
    def get_all_questions():
        """Return all quiz questions organized by difficulty"""
        return {
            "Beginner": QuizDatabase._get_beginner_questions(),
            "Intermediate": QuizDatabase._get_intermediate_questions(),
            "Advanced": QuizDatabase._get_advanced_questions()
        }
    
    @staticmethod
    def _get_beginner_questions():
        """Beginner level questions about fundamental airfoil concepts"""
        return [
            QuizQuestion(
                question="What does NACA stand for?",
                options=[
                    "National Advisory Committee for Aeronautics",
                    "North American Civil Aviation",
                    "National Aircraft Control Authority",
                    "Naval Air Combat Academy"
                ],
                correct_index=0,
                explanation="NACA stands for National Advisory Committee for Aeronautics, the predecessor to NASA. It was established in 1915 and conducted extensive research on airfoils and aircraft design."
            ),
            QuizQuestion(
                question="In a NACA 2412 airfoil, what does the '24' represent?",
                options=[
                    "Maximum thickness as 24% of chord",
                    "2% camber at 40% chord position",
                    "24% camber",
                    "Wing area of 24 square feet"
                ],
                correct_index=1,
                explanation="In NACA 2412: '2' = 2% maximum camber, '4' = camber position at 40% chord, '12' = 12% maximum thickness. The first two digits together (24) indicate 2% camber at 40% chord position."
            ),
            QuizQuestion(
                question="What is the primary function of an airfoil?",
                options=[
                    "To reduce aircraft weight",
                    "To generate lift by creating pressure difference",
                    "To store fuel",
                    "To provide structural support"
                ],
                correct_index=1,
                explanation="An airfoil's primary function is to generate lift by creating a pressure difference between its upper and lower surfaces. The curved shape deflects air and creates lower pressure on top."
            ),
            QuizQuestion(
                question="What does a negative pressure coefficient (Cp) indicate?",
                options=[
                    "High pressure region",
                    "Low pressure region (suction)",
                    "Zero velocity",
                    "Stagnation point"
                ],
                correct_index=1,
                explanation="A negative Cp indicates pressure below atmospheric pressure (suction), typically found on the upper surface of an airfoil where air accelerates and pressure drops."
            ),
            QuizQuestion(
                question="A NACA 0012 airfoil is:",
                options=[
                    "Cambered with 12% thickness",
                    "Symmetric with 12% thickness",
                    "Cambered with 1.2% thickness",
                    "Symmetric with 1.2% thickness"
                ],
                correct_index=1,
                explanation="NACA 0012 is symmetric (first digit 0 = no camber) with 12% maximum thickness ratio. Symmetric airfoils perform identically when inverted, making them ideal for aerobatic aircraft."
            ),
            QuizQuestion(
                question="Which part of an airfoil typically generates the most lift?",
                options=[
                    "Leading edge",
                    "Trailing edge",
                    "Upper surface",
                    "Lower surface"
                ],
                correct_index=2,
                explanation="The upper surface typically generates most of the lift through suction (low pressure). While the lower surface contributes through positive pressure, the upper surface's contribution is usually larger."
            ),
            QuizQuestion(
                question="What is the chord line of an airfoil?",
                options=[
                    "The curved upper surface",
                    "The curved lower surface",
                    "A straight line from leading edge to trailing edge",
                    "The camber line"
                ],
                correct_index=2,
                explanation="The chord line is a straight line connecting the leading edge to the trailing edge of an airfoil. It serves as a reference for measuring camber and angle of attack."
            )
        ]
    
    @staticmethod
    def _get_intermediate_questions():
        """Intermediate level questions about airfoil design and applications"""
        return [
            QuizQuestion(
                question="What is the primary advantage of a symmetric airfoil (NACA 00XX)?",
                options=[
                    "Higher maximum lift coefficient",
                    "Better fuel efficiency",
                    "Identical performance when inverted",
                    "Lower manufacturing cost"
                ],
                correct_index=2,
                explanation="Symmetric airfoils perform identically when inverted, making them ideal for aerobatic aircraft that frequently fly inverted. They have zero pitching moment at zero lift."
            ),
            QuizQuestion(
                question="How does increasing airfoil thickness generally affect performance?",
                options=[
                    "Increases lift and decreases drag",
                    "Provides structural strength but increases drag",
                    "Decreases both lift and drag",
                    "Has no effect on performance"
                ],
                correct_index=1,
                explanation="Thicker airfoils provide more structural strength and internal volume but typically increase drag, especially at higher speeds. There's always a trade-off between structure and aerodynamic efficiency."
            ),
            QuizQuestion(
                question="Why do commercial aircraft typically use cambered airfoils rather than symmetric ones?",
                options=[
                    "Lower manufacturing cost",
                    "Better lift at cruise conditions",
                    "Easier maintenance",
                    "Better high-speed performance"
                ],
                correct_index=1,
                explanation="Cambered airfoils generate lift more efficiently at positive angles of attack used in normal flight. They provide better lift-to-drag ratios at cruise conditions compared to symmetric airfoils."
            ),
            QuizQuestion(
                question="Which airfoil characteristic is most important for high-speed flight?",
                options=[
                    "High camber for maximum lift",
                    "Large thickness for strength",
                    "Low thickness to delay shock formation",
                    "Blunt leading edge for stability"
                ],
                correct_index=2,
                explanation="Thin airfoils delay the onset of compressibility effects and shock wave formation at high speeds. This is crucial for aircraft operating at transonic and supersonic speeds."
            ),
            QuizQuestion(
                question="What is the primary reason wind turbine blades use high-camber airfoils?",
                options=[
                    "Structural simplicity",
                    "Maximum power extraction from wind",
                    "Reduced noise generation",
                    "Lower manufacturing cost"
                ],
                correct_index=1,
                explanation="High-camber airfoils maximize the lift coefficient and power extraction efficiency from wind. Wind turbines need to extract maximum energy from the available wind resource."
            )
        ]
    
    @staticmethod
    def _get_advanced_questions():
        """Advanced level questions about complex aerodynamic phenomena"""
        return [
            QuizQuestion(
                question="In transonic flow over an airfoil, what causes wave drag?",
                options=[
                    "Viscous friction in the boundary layer",
                    "Shock wave formation and compression losses",
                    "Induced drag from finite span effects",
                    "Surface roughness interactions"
                ],
                correct_index=1,
                explanation="Wave drag results from shock wave formation in transonic flow, where compression losses occur across shocks. This becomes significant as local flow velocities approach and exceed the speed of sound."
            ),
            QuizQuestion(
                question="What is the primary purpose of supercritical airfoil design?",
                options=[
                    "Increase maximum lift coefficient",
                    "Delay shock formation to higher Mach numbers",
                    "Reduce structural weight",
                    "Improve low-speed handling"
                ],
                correct_index=1,
                explanation="Supercritical airfoils are designed with flatter upper surfaces to delay shock formation, allowing higher cruise Mach numbers before significant wave drag develops."
            ),
            QuizQuestion(
                question="How does adverse pressure gradient affect boundary layer behavior?",
                options=[
                    "Accelerates the flow",
                    "Has no effect on boundary layer",
                    "Promotes flow separation",
                    "Reduces skin friction"
                ],
                correct_index=2,
                explanation="Adverse pressure gradients (increasing pressure in flow direction) decelerate the boundary layer flow and can lead to separation, especially in regions of high curvature or at high angles of attack."
            ),
            QuizQuestion(
                question="What is the primary limitation of inviscid flow analysis for airfoils?",
                options=[
                    "Cannot predict lift accurately",
                    "Cannot predict boundary layer effects and separation",
                    "Too computationally expensive",
                    "Cannot handle compressible flow"
                ],
                correct_index=1,
                explanation="Inviscid analysis cannot capture boundary layer effects, flow separation, or drag prediction accurately. It provides good lift predictions but misses viscous phenomena that are crucial for complete airfoil analysis."
            ),
            QuizQuestion(
                question="What is the Kutta condition and why is it important?",
                options=[
                    "A manufacturing tolerance specification",
                    "A condition ensuring smooth flow departure at the trailing edge",
                    "A structural load requirement",
                    "A noise generation criterion"
                ],
                correct_index=1,
                explanation="The Kutta condition requires smooth flow departure at the trailing edge, which determines the circulation around the airfoil and thus the lift. Without it, theoretical solutions would be non-unique."
            ),
            QuizQuestion(
                question="In the context of airfoil design, what is the critical Mach number?",
                options=[
                    "The Mach number at which the aircraft stalls",
                    "The Mach number at which sonic flow first occurs locally",
                    "The maximum operating Mach number",
                    "The Mach number for minimum drag"
                ],
                correct_index=1,
                explanation="The critical Mach number is when sonic flow (M=1) first occurs locally on the airfoil surface, typically on the upper surface. Beyond this point, shock waves begin to form and wave drag increases."
            )
        ]

class QuizSession:
    """Manages the state of a quiz session"""
    
    def __init__(self, level: str):
        self.level = level
        self.questions = QuizDatabase.get_all_questions()[level][:UI_CFG.QUESTIONS_PER_LEVEL]
        self.current_question_index = 0
        self.answers = []
        self.start_time = datetime.now()
        self.end_time = None
        self.is_completed = False
    
    @property
    def current_question(self) -> QuizQuestion:
        """Get the current question"""
        if self.current_question_index < len(self.questions):
            return self.questions[self.current_question_index]
        return None
    
    @property
    def total_questions(self) -> int:
        """Get total number of questions"""
        return len(self.questions)
    
    @property
    def score(self) -> int:
        """Get current score (number of correct answers)"""
        return sum(1 for answer in self.answers if answer['is_correct'])
    
    @property
    def percentage(self) -> float:
        """Get current percentage score"""
        if len(self.answers) == 0:
            return 0.0
        return (self.score / len(self.answers)) * 100
    
    @property
    def final_percentage(self) -> float:
        """Get final percentage based on total questions"""
        if self.total_questions == 0:
            return 0.0
        return (self.score / self.total_questions) * 100
    
    @property
    def grade(self) -> str:
        """Get letter grade based on final percentage"""
        final_perc = self.final_percentage
        if final_perc >= UI_CFG.EXCELLENT_SCORE:
            return "Excellent"
        elif final_perc >= UI_CFG.PASSING_SCORE:
            return "Good"
        else:
            return "Keep Learning"
    
    def submit_answer(self, selected_index: int):
        """Submit an answer for the current question"""
        if self.current_question is None:
            raise ValueError("No current question available")
        
        question = self.current_question
        is_correct = question.is_correct(selected_index)
        
        answer = {
            'question': question.question,
            'options': question.options,
            'selected_index': selected_index,
            'selected_answer': question.options[selected_index],
            'correct_index': question.correct_index,
            'correct_answer': question.correct_answer,
            'is_correct': is_correct,
            'explanation': question.explanation
        }
        
        self.answers.append(answer)
        self.current_question_index += 1
        
        # Check if quiz is completed
        if self.current_question_index >= self.total_questions:
            self.is_completed = True
            self.end_time = datetime.now()
        
        return answer
    
    def get_duration(self) -> str:
        """Get quiz duration as formatted string"""
        end = self.end_time if self.end_time else datetime.now()
        duration = end - self.start_time
        minutes = int(duration.total_seconds() // 60)
        seconds = int(duration.total_seconds() % 60)
        return f"{minutes}m {seconds}s"

class QuizController:
    """Main controller for the quiz system"""
    
    def __init__(self):
        self._initialize_quiz_state()
    
    def _initialize_quiz_state(self):
        """Initialize quiz-related session state"""
        if 'quiz_session' not in st.session_state:
            st.session_state.quiz_session = None
        if 'quiz_level' not in st.session_state:
            st.session_state.quiz_level = None
    
    def run_quiz_system(self):
        """Main quiz system workflow"""
        st.markdown("## Airfoil Knowledge Quiz")
        st.markdown("Test your understanding of airfoil aerodynamics and design principles!")
        st.markdown("---")
        
        # Get current session
        current_session = st.session_state.quiz_session
        
        # Route to appropriate quiz state
        if current_session is None:
            self._handle_quiz_setup()
        elif not current_session.is_completed:
            self._handle_active_quiz(current_session)
        else:
            self._handle_completed_quiz(current_session)
    
    def _handle_quiz_setup(self):
        """Handle quiz setup and level selection"""
        col_setup1, col_setup2 = st.columns([2, 1])
        
        with col_setup1:
            st.markdown("### Choose Your Challenge Level")
            
            # Level descriptions
            level_descriptions = {
                "Beginner": "Fundamental concepts about airfoils and basic aerodynamics",
                "Intermediate": "Applied aerodynamics and airfoil design principles", 
                "Advanced": "Complex aerodynamic phenomena and advanced design concepts"
            }
            
            selected_level = st.selectbox(
                "**Select Quiz Difficulty**",
                UI_CFG.DIFFICULTY_LEVELS,
                index=0,
                help="Choose the difficulty level that matches your knowledge"
            )
            
            st.info(f"**{selected_level} Level**: {level_descriptions[selected_level]}")
            
            # Preview questions for selected level
            questions = QuizDatabase.get_all_questions()[selected_level]
            st.markdown(f"**Sample Topic:** {questions[0].question}")
        
        with col_setup2:
            st.markdown("### Quiz Format")
            st.markdown(f"- **{UI_CFG.QUESTIONS_PER_LEVEL} Questions** per level")
            st.markdown("- **Multiple Choice** format")
            st.markdown("- **Instant feedback** with explanations")
            st.markdown("- **Performance tracking**")
            st.markdown("- **Progressive difficulty**")
        
        # Start button
        col_start1, col_start2, col_start3 = st.columns([1, 2, 1])
        with col_start2:
            if st.button(f"Start {selected_level} Quiz", type="primary", use_container_width=True):
                st.session_state.quiz_session = QuizSession(selected_level)
                st.session_state.quiz_level = selected_level
                st.rerun()
    
    def _handle_active_quiz(self, session: QuizSession):
        """Handle active quiz session"""
        # Progress indicator
        progress_text = f"Question {session.current_question_index + 1} of {session.total_questions}"
        progress_value = session.current_question_index / session.total_questions
        st.progress(progress_value, text=progress_text)
        
        # Current score display
        if session.answers:
            col_score1, col_score2, col_score3 = st.columns([1, 1, 2])
            with col_score1:
                st.metric("Current Score", f"{session.score}/{len(session.answers)}")
            with col_score2:
                st.metric("Percentage", f"{session.percentage:.1f}%")
            with col_score3:
                st.metric("Time Elapsed", session.get_duration())
        
        st.markdown("---")
        
        # Display current question
        current_question = session.current_question
        if current_question is None:
            st.error("No question available")
            return
        
        st.markdown(f"<h2>Question {session.current_question_index + 1}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='font-size: 18px;'>{current_question.question}</h3>", unsafe_allow_html=True)
        
        # Answer options
        selected_answer = st.radio(
            "Select your answer:",
            range(len(current_question.options)),
            format_func=lambda x: current_question.options[x],
            key=f"quiz_q_{session.current_question_index}_{session.level}"
        )
        
        # Submit button
        col_submit1, col_submit2 = st.columns([1, 3])
        with col_submit1:
            submit_clicked = st.button("Submit Answer", type="primary")
        
        if submit_clicked:
            # Submit answer and show feedback
            answer = session.submit_answer(selected_answer)
            self._show_immediate_feedback(answer)
            
            # Check if quiz is completed
            if session.is_completed:
                st.success("Quiz completed! Scroll down to see your results.")
            else:
                st.info("Answer submitted! Next question loading...")
            
            st.rerun()
    
    def _show_immediate_feedback(self, answer: dict):
        """Show immediate feedback after answer submission"""
        if answer['is_correct']:
            st.success(f"Correct! {answer['correct_answer']}")
        else:
            st.error(f"Incorrect. The correct answer is: {answer['correct_answer']}")
        
        # Show explanation
        with st.expander("Explanation", expanded=True):
            st.info(answer['explanation'])
    
    def _handle_completed_quiz(self, session: QuizSession):
        """Handle completed quiz display and options"""
        st.markdown("## Quiz Completed!")
        
        # Score summary
        col_final1, col_final2, col_final3, col_final4 = st.columns(4)
        
        with col_final1:
            st.metric("Final Score", f"{session.score}/{session.total_questions}")
        with col_final2:
            st.metric("Percentage", f"{session.final_percentage:.1f}%")
        with col_final3:
            st.metric("Grade", session.grade)
        with col_final4:
            st.metric("Duration", session.get_duration())
        
        # Performance feedback
        self._display_performance_feedback(session)
        
        # Performance visualization
        self._create_performance_chart(session)
        
        # Detailed results
        self._display_detailed_results(session)
        
        # Restart options
        self._create_restart_options(session)
    
    def _display_performance_feedback(self, session: QuizSession):
        """Display personalized performance feedback"""
        percentage = session.final_percentage
        grade = session.grade
        
        if percentage >= UI_CFG.EXCELLENT_SCORE:
            st.success(f"Outstanding performance! You achieved {percentage:.1f}% on the {session.level} level.")
            feedback = "You demonstrate excellent understanding of airfoil concepts. Consider advancing to the next difficulty level!"
        elif percentage >= UI_CFG.PASSING_SCORE:
            st.info(f"Good work! You scored {percentage:.1f}%, demonstrating solid understanding of airfoil concepts.")
            feedback = "You have a good grasp of the fundamentals. Review the explanations below to strengthen your knowledge further."
        else:
            st.warning(f"You scored {percentage:.1f}%. Don't get discouraged - airfoil aerodynamics can be challenging!")
            feedback = "Focus on reviewing the detailed explanations below. Consider retaking the quiz to reinforce your learning."
        
        st.markdown(f"**Feedback:** {feedback}")
        
        # Suggest next steps
        self._suggest_next_steps(session)
    
    def _suggest_next_steps(self, session: QuizSession):
        """Suggest next steps based on performance"""
        levels = UI_CFG.DIFFICULTY_LEVELS
        current_index = levels.index(session.level) if session.level in levels else 0
        
        if session.final_percentage >= UI_CFG.EXCELLENT_SCORE and current_index < len(levels) - 1:
            next_level = levels[current_index + 1]
            st.info(f"Ready for a challenge? Try the {next_level} level!")
        elif session.final_percentage >= UI_CFG.PASSING_SCORE:
            if current_index < len(levels) - 1:
                next_level = levels[current_index + 1]
                st.info(f"When you're ready, consider advancing to the {next_level} level.")
            else:
                st.success("You've mastered the advanced level! Excellent work!")
        else:
            st.info(f"Practice more with {session.level} level questions to build confidence before advancing.")
    
    def _create_performance_chart(self, session: QuizSession):
        """Create performance visualization"""
        if len(session.answers) < 3:
            return
        
        st.markdown("### Performance Progress")
        
        # Calculate cumulative performance
        question_numbers = list(range(1, len(session.answers) + 1))
        cumulative_percentage = []
        
        running_score = 0
        for i, answer in enumerate(session.answers):
            if answer['is_correct']:
                running_score += 1
            cumulative_percentage.append((running_score / (i + 1)) * 100)
        
        # Create chart
        fig, ax = plt.subplots(figsize=(7, 3.5))
        
        ax.plot(question_numbers, cumulative_percentage, 
                marker='o', linewidth=3, markersize=8, color='#1f77b4')
        ax.axhline(y=UI_CFG.PASSING_SCORE, color='green', 
                  linestyle='--', alpha=0.7, label=f'Passing ({UI_CFG.PASSING_SCORE}%)')
        ax.axhline(y=UI_CFG.EXCELLENT_SCORE, color='gold', 
                  linestyle='--', alpha=0.7, label=f'Excellent ({UI_CFG.EXCELLENT_SCORE}%)')
        
        ax.set_xlabel('Question Number')
        ax.set_ylabel('Cumulative Score (%)')
        ax.set_title(f'Quiz Performance Progress - {session.level} Level')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 100)
        
        # Add final score annotation
        final_score = cumulative_percentage[-1]
        ax.annotate(f'Final: {final_score:.1f}%', 
                   xy=(len(question_numbers), final_score),
                   xytext=(len(question_numbers)-0.5, final_score+10),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   fontsize=12, ha='center')
        
        st.pyplot(fig)
        PlotUtilities.cleanup_figures()
    
    def _display_detailed_results(self, session: QuizSession):
        """Display detailed results with explanations"""
        st.markdown("### Detailed Results & Learning")
        
        for i, answer in enumerate(session.answers):
            with st.expander(f"Question {i+1}: {answer['question'][:50]}..."):
                st.markdown(f"**Question:** {answer['question']}")
                
                # Show answer status
                if answer['is_correct']:
                    st.success(f"✅ Your Answer: {answer['selected_answer']} (Correct!)")
                else:
                    st.error(f"❌ Your Answer: {answer['selected_answer']} (Incorrect)")
                    st.success(f"✅ Correct Answer: {answer['correct_answer']}")
                
                # Show all options with indicators
                st.markdown("**All Options:**")
                for j, option in enumerate(answer['options']):
                    if j == answer['correct_index']:
                        st.markdown(f"✅ {option}")
                    elif j == answer['selected_index'] and not answer['is_correct']:
                        st.markdown(f"❌ {option}")
                    else:
                        st.markdown(f"⚪ {option}")
                
                # Show explanation
                st.markdown("**Explanation:**")
                st.info(answer['explanation'])
    
    def _create_restart_options(self, session: QuizSession):
        """Create options for restarting or advancing quiz"""
        st.markdown("---")
        st.markdown("### What's Next?")
        
        col_restart1, col_restart2, col_restart3 = st.columns(3)
        
        with col_restart1:
            if st.button(f"Retake {session.level} Quiz", use_container_width=True):
                st.session_state.quiz_session = QuizSession(session.level)
                st.rerun()
        
        with col_restart2:
            # Next level button if applicable
            levels = UI_CFG.DIFFICULTY_LEVELS
            current_index = levels.index(session.level) if session.level in levels else 0
            
            if (session.final_percentage >= UI_CFG.PASSING_SCORE and 
                current_index < len(levels) - 1):
                next_level = levels[current_index + 1]
                if st.button(f"Try {next_level} Level", use_container_width=True):
                    st.session_state.quiz_session = QuizSession(next_level)
                    st.session_state.quiz_level = next_level
                    st.rerun()
        
        with col_restart3:
            if st.button("Choose Different Level", use_container_width=True):
                st.session_state.quiz_session = None
                st.session_state.quiz_level = None
                st.rerun()

# ============================================================================
# PHASE 8: MAIN APPLICATION ENTRY POINT
# ============================================================================

def main():
    """Main application function with complete functionality"""
    
    # Apply plot styling
    PlotUtilities.apply_style_theme()
    
    # Initialize session state
    PageModeManager.initialize_session_state()
    
    # Load model at startup
    model_manager, model_loaded = get_model_manager()
    
    # Create app header
    UIComponents.create_app_header()
    
    # Create navigation
    single_mode, multiple_mode, quiz_mode = UIComponents.create_navigation_buttons()
    
    # Handle navigation
    current_mode = PageModeManager.handle_navigation(single_mode, multiple_mode, quiz_mode)
    
    # Display current mode information
    UIComponents.display_current_mode_info(current_mode)
    
    # Route to appropriate analysis function
    try:
        if current_mode == 'single':
            if ErrorHandler.handle_model_loading_error(model_manager, model_loaded):
                analyzer = AnalysisController(model_manager)
                analyzer.run_single_analysis()
        
        elif current_mode == 'multiple':
            if ErrorHandler.handle_model_loading_error(model_manager, model_loaded):
                analyzer = MultipleAirfoilAnalyzer(model_manager)
                analyzer.run_analysis()

        elif current_mode == 'quiz':
            quiz_controller = QuizController()
            quiz_controller.run_quiz_system()
    
    except Exception as e:
        ErrorHandler.handle_prediction_error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please refresh the page and try again.")
    
    finally:
        # Clean up resources
        PlotUtilities.cleanup_figures()
    
    # Add footer with additional information
    create_app_footer()

def create_app_footer():
    """Create application footer with documentation and information"""
    st.markdown("---")
    st.markdown("## Application Guide & Resources")
    
    col_doc1, col_doc2 = st.columns(2)
    
    with col_doc1:
        create_usage_guide()
    
    with col_doc2:
        create_technical_info()
    
    create_about_section()

def create_usage_guide():
    """Create usage guide section"""
    with st.expander("How to Use This Application"):
        st.markdown("""
        ### Single Airfoil Analysis
        1. **Choose an aircraft preset** or select "Custom" for manual configuration
        2. **Adjust NACA parameters** using the sidebar sliders
        3. **Preview the airfoil geometry** to verify your configuration
        4. **Click "Predict Cp Distribution"** to run the analysis
        5. **Export results** as CSV, text report, or comprehensive PDF
        
        ### Multiple Airfoil Comparison
        1. **Select number of airfoils** to compare (2-4 airfoils)
        2. **Choose quick preset comparisons** or configure manually
        3. **Set up each airfoil** using the configuration panels
        4. **Click "Analyze All Airfoils"** to run the comparison
        5. **Review comparative visualizations** and performance rankings
        6. **Export comparison data** and comprehensive PDF reports
        7. **Analyze design trade-offs** from the engineering insights
        
        ### Airfoil Knowledge Quiz
        1. **Select your difficulty level** (Beginner/Intermediate/Advanced)
        2. **Answer multiple-choice questions** about airfoil concepts
        3. **Get instant feedback** with detailed explanations
        4. **Review your performance** and progress tracking
        5. **Advance to higher difficulty levels** based on your score
        6. **Learn from detailed explanations** for each question
        
        ### Understanding NACA Codes
        **NACA 2412 Example:**
        - **2** = 2% maximum camber
        - **4** = Camber position at 40% chord
        - **12** = 12% maximum thickness
        
        ### Interpreting Results
        - **Negative Cp values** = Suction (pressure below atmospheric)
        - **Positive Cp values** = Compression (pressure above atmospheric)
        - **Peak suction** = Most negative Cp value (critical for lift generation)
        """)

def create_technical_info():
    """Create technical information section"""
    with st.expander("Technical Information"):
        st.markdown("""
        ### Neural Network Model
        - **Architecture**: Deep feedforward neural network
        - **Training Data**: 1000+ NACA airfoil configurations
        - **Validation**: Cross-validated on 300+ test cases
        - **Accuracy**: 95%+ correlation with CFD results
        
        ### Analysis Conditions
        - **Flow Type**: Inviscid, incompressible flow
        - **Angle of Attack**: Zero degrees
        - **Mach Number**: Subsonic (M < 0.8)
        - **Reynolds Number**: High Reynolds number assumptions
        
        ### Limitations
        - NACA 4-digit airfoils only
        - Zero angle of attack only
        - Inviscid flow assumptions
        - No boundary layer effects
        
        ### System Requirements
        ```
        Python 3.7+
        streamlit>=1.28.0
        tensorflow>=2.10.0
        numpy, pandas, matplotlib, scipy
        reportlab (optional, for PDF reports)
        ```
        """)

def create_about_section():
    """Create about section"""
    with st.expander("About This Application"):
        st.markdown("""
        ### Purpose
        This application provides an interactive platform for analyzing NACA 4-digit airfoils using 
        machine learning predictions. It serves educational, research, and preliminary design purposes.
        
        ### Target Users
        - **Students** learning aerodynamics and airfoil design
        - **Engineers** performing preliminary airfoil analysis
        - **Researchers** comparing airfoil characteristics
        - **Educators** teaching aerodynamic concepts
        
        ### Key Features
        - **Real-time predictions** using trained neural networks
        - **Interactive visualizations** with detailed analysis
        - **Professional PDF reports** with comprehensive results
        - **Aircraft preset configurations** for common applications
        - **Export capabilities** in multiple formats
        
        ### Model Training
        The neural network was trained on a comprehensive database of CFD results for NACA 4-digit 
        airfoils, covering the full parameter space of camber (0-7%), position (20-60%), and thickness (6-30%).
        
        ### Future Enhancements
        - Support for 5-digit NACA airfoils
        - Angle of attack variations
        - Reynolds number effects
        - Compressibility corrections
        - Custom airfoil coordinate uploads
        - 3D visualization capabilities
        
        ### Version Information
        - **Current Version**: 1.0.0
        - **Last Updated**: 2025
        - **Model Version**: Neural Network v2.1
        
        ### Support & Feedback
        For technical support, feature requests, or educational partnerships, 
        please refer to the documentation or contact the development team.

        ### Developed using Streamlit, TensorFlow, and Keras
        """)

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Initialize the application
    try:
        main()
    except Exception as e:
        st.error("Critical application error. Please refresh the page.")
        st.exception(e)

