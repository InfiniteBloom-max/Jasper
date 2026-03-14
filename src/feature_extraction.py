"""
JASPER: Japanese x Sri Lankan Textile Design Analysis
Feature Extraction Module

Extracts 44 quantitative features from textile images:
- Color features (dominant palettes, RGB profiles, warm/cool scores)
- Pattern complexity (edge density, entropy, Canny metrics)
- Texture features (LBP, contrast, Haralick metrics)
- Symmetry and geometric features
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


class TextileFeatureExtractor:
    """Extract comprehensive design DNA features from textile images"""
    
    def __init__(self, n_colors=5, lbp_radius=3, lbp_points=24):
        """
        Initialize feature extractor
        
        Args:
            n_colors: Number of dominant colors to extract (default: 5)
            lbp_radius: Radius for Local Binary Pattern (default: 3)
            lbp_points: Number of points for LBP (default: 24)
        """
        self.n_colors = n_colors
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points
        
    def extract_all_features(self, image_path):
        """
        Extract all 44 features from a single textile image
        
        Args:
            image_path: Path to textile image
            
        Returns:
            dict: Dictionary containing all extracted features
        """
        # Load image
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        features = {}
        
        # Color features
        features.update(self._extract_color_features(img_rgb))
        
        # Pattern complexity features
        features.update(self._extract_pattern_features(img_gray))
        
        # Texture features
        features.update(self._extract_texture_features(img_gray))
        
        # Symmetry features
        features.update(self._extract_symmetry_features(img_gray))
        
        # Geometric features
        features.update(self._extract_geometric_features(img_gray))
        
        return features
    
    def _extract_color_features(self, img_rgb):
        """Extract color-based features using k-means clustering"""
        features = {}
        
        # Reshape for k-means
        pixels = img_rgb.reshape(-1, 3)
        
        # K-means clustering for dominant colors
        kmeans = KMeans(n_clusters=self.n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get dominant colors and their frequencies
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        
        # Sort by frequency
        unique, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(-counts)
        
        # Store top 5 dominant colors
        for i in range(self.n_colors):
            idx = sorted_indices[i] if i < len(sorted_indices) else sorted_indices[0]
            color = centers[idx]
            features[f'dominant_color_{i+1}_r'] = float(color[0])
            features[f'dominant_color_{i+1}_g'] = float(color[1])
            features[f'dominant_color_{i+1}_b'] = float(color[2])
            features[f'dominant_color_{i+1}_freq'] = float(counts[idx] / len(labels))
        
        # Average RGB values
        features['avg_r'] = float(np.mean(img_rgb[:, :, 0]))
        features['avg_g'] = float(np.mean(img_rgb[:, :, 1]))
        features['avg_b'] = float(np.mean(img_rgb[:, :, 2]))
        
        # Color variance (spread of colors)
        features['color_variance'] = float(np.mean(np.var(pixels, axis=0)))
        
        # Warm/Cool score (critical metric from paper)
        # Warm colors: reds, oranges, yellows (R > B, R > G)
        # Cool colors: blues, greens (B > R or G > R)
        warm_pixels = np.sum((pixels[:, 0] > pixels[:, 2]) & (pixels[:, 0] > pixels[:, 1]))
        cool_pixels = np.sum((pixels[:, 2] > pixels[:, 0]) | (pixels[:, 1] > pixels[:, 0]))
        total_pixels = len(pixels)
        features['warm_cool_score'] = float((warm_pixels - cool_pixels) / total_pixels)
        
        # Color saturation (HSV)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        features['avg_saturation'] = float(np.mean(img_hsv[:, :, 1]))
        features['avg_value'] = float(np.mean(img_hsv[:, :, 2]))
        
        return features
    
    def _extract_pattern_features(self, img_gray):
        """Extract pattern complexity features"""
        features = {}
        
        # Edge detection using Canny
        edges = cv2.Canny(img_gray, 100, 200)
        features['edge_density'] = float(np.sum(edges > 0) / edges.size)
        
        # Sobel edge detection for directional analysis
        sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        features['gradient_magnitude'] = float(np.mean(gradient_magnitude))
        features['gradient_std'] = float(np.std(gradient_magnitude))
        
        # Shannon entropy (pattern complexity)
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        features['shannon_entropy'] = float(entropy(hist))
        
        # Pattern density (high frequency content)
        f_transform = np.fft.fft2(img_gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # High frequency energy
        rows, cols = img_gray.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), dtype=np.uint8)
        r = 30
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
        mask[mask_area] = 0
        
        high_freq_energy = np.sum(magnitude_spectrum * mask)
        total_energy = np.sum(magnitude_spectrum)
        features['high_freq_ratio'] = float(high_freq_energy / (total_energy + 1e-10))
        
        return features
    
    def _extract_texture_features(self, img_gray):
        """Extract texture features using LBP and Haralick"""
        features = {}
        
        # Local Binary Pattern (LBP)
        lbp = local_binary_pattern(img_gray, self.lbp_points, self.lbp_radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, self.lbp_points + 3), range=(0, self.lbp_points + 2))
        lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()
        
        features['lbp_uniformity'] = float(np.sum(lbp_hist**2))
        features['lbp_entropy'] = float(entropy(lbp_hist))
        features['lbp_energy'] = float(np.sum(lbp_hist**2))
        
        # Haralick texture features using GLCM
        # Normalize image for GLCM (required: 0-255 uint8)
        glcm = graycomatrix(img_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                           levels=256, symmetric=True, normed=True)
        
        # Extract Haralick features (averaged across angles)
        features['glcm_contrast'] = float(np.mean(graycoprops(glcm, 'contrast')))
        features['glcm_dissimilarity'] = float(np.mean(graycoprops(glcm, 'dissimilarity')))
        features['glcm_homogeneity'] = float(np.mean(graycoprops(glcm, 'homogeneity')))
        features['glcm_energy'] = float(np.mean(graycoprops(glcm, 'energy')))
        features['glcm_correlation'] = float(np.mean(graycoprops(glcm, 'correlation')))
        features['glcm_asm'] = float(np.mean(graycoprops(glcm, 'ASM')))
        
        # Texture complexity (combination metric)
        features['texture_complexity'] = float(
            features['glcm_contrast'] * features['lbp_entropy']
        )
        
        # Texture contrast (critical metric from paper)
        features['texture_contrast'] = features['glcm_contrast']
        
        return features
    
    def _extract_symmetry_features(self, img_gray):
        """Extract symmetry features"""
        features = {}
        
        # Vertical symmetry
        left_half = img_gray[:, :img_gray.shape[1]//2]
        right_half = cv2.flip(img_gray[:, img_gray.shape[1]//2:], 1)
        
        # Ensure same size
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        if left_half.shape == right_half.shape:
            vertical_diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
            features['vertical_symmetry'] = float(1.0 / (1.0 + vertical_diff / 255.0))
        else:
            features['vertical_symmetry'] = 0.0
        
        # Horizontal symmetry
        top_half = img_gray[:img_gray.shape[0]//2, :]
        bottom_half = cv2.flip(img_gray[img_gray.shape[0]//2:, :], 0)
        
        min_height = min(top_half.shape[0], bottom_half.shape[0])
        top_half = top_half[:min_height, :]
        bottom_half = bottom_half[:min_height, :]
        
        if top_half.shape == bottom_half.shape:
            horizontal_diff = np.mean(np.abs(top_half.astype(float) - bottom_half.astype(float)))
            features['horizontal_symmetry'] = float(1.0 / (1.0 + horizontal_diff / 255.0))
        else:
            features['horizontal_symmetry'] = 0.0
        
        # Overall symmetry score (critical metric from paper)
        features['symmetry_score'] = float((features['vertical_symmetry'] + features['horizontal_symmetry']) / 2.0)
        
        return features
    
    def _extract_geometric_features(self, img_gray):
        """Extract geometric and motif features"""
        features = {}
        
        # Detect edges for shape analysis
        edges = cv2.Canny(img_gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Geometric ratios
            areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 100]
            
            if len(areas) > 0:
                features['avg_motif_area'] = float(np.mean(areas))
                features['motif_area_std'] = float(np.std(areas))
                features['num_motifs'] = len(areas)
                
                # Aspect ratios of bounding rectangles
                aspect_ratios = []
                for c in contours:
                    if cv2.contourArea(c) > 100:
                        x, y, w, h = cv2.boundingRect(c)
                        if h > 0:
                            aspect_ratios.append(w / h)
                
                if len(aspect_ratios) > 0:
                    features['avg_aspect_ratio'] = float(np.mean(aspect_ratios))
                    features['aspect_ratio_std'] = float(np.std(aspect_ratios))
                else:
                    features['avg_aspect_ratio'] = 1.0
                    features['aspect_ratio_std'] = 0.0
            else:
                features['avg_motif_area'] = 0.0
                features['motif_area_std'] = 0.0
                features['num_motifs'] = 0
                features['avg_aspect_ratio'] = 1.0
                features['aspect_ratio_std'] = 0.0
        else:
            features['avg_motif_area'] = 0.0
            features['motif_area_std'] = 0.0
            features['num_motifs'] = 0
            features['avg_aspect_ratio'] = 1.0
            features['aspect_ratio_std'] = 0.0
        
        # Circularity (4π*area/perimeter²)
        circularities = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 100:
                perimeter = cv2.arcLength(c, True)
                if perimeter > 0:
                    circularities.append(4 * np.pi * area / (perimeter ** 2))
        
        features['avg_circularity'] = float(np.mean(circularities)) if len(circularities) > 0 else 0.0
        
        return features


def extract_dataset_features(image_paths, labels, output_csv=None, verbose=True):
    """
    Extract features from entire dataset
    
    Args:
        image_paths: List of image file paths
        labels: List of labels (e.g., 'japanese_textiles', 'sri_lankan_textiles')
        output_csv: Optional path to save CSV
        verbose: Print progress
        
    Returns:
        pandas.DataFrame with all features
    """
    import pandas as pd
    from tqdm import tqdm
    
    extractor = TextileFeatureExtractor()
    results = []
    
    iterator = tqdm(zip(image_paths, labels), total=len(image_paths)) if verbose else zip(image_paths, labels)
    
    for img_path, label in iterator:
        try:
            features = extractor.extract_all_features(img_path)
            features['filepath'] = str(img_path)
            features['label'] = label
            results.append(features)
        except Exception as e:
            if verbose:
                print(f"Error processing {img_path}: {e}")
            continue
    
    df = pd.DataFrame(results)
    
    if output_csv:
        df.to_csv(output_csv, index=False)
        if verbose:
            print(f"\nSaved features to {output_csv}")
    
    return df
