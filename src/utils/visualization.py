import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_heatmaps(image, heatmaps, alpha=0.5):
    """
    Visualize keypoint heatmaps overlaid on the original image.
    
    Args:
        image: Original image (H, W, 3)
        heatmaps: Heatmaps tensor (K+1, H, W)
        alpha: Transparency of heatmap overlay
    
    Returns:
        Visualization image with heatmap overlay
    """
    # Skip background channel (index 0)
    combined_heatmap = np.max(heatmaps[1:], axis=0)
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(
        combined_heatmap,
        (image.shape[1], image.shape[0])
    )
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    
    # Overlay heatmap on image
    return cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

def visualize_keypoints(image, points, homography, radius=5, color=(0, 0, 255)):
    """
    Visualize keypoints on the original image.
    
    Args:
        image: Original image
        points: List of (x, y) points in court coordinates
        homography: Homography matrix
        radius: Radius of keypoint circles
        color: Color of keypoint circles (BGR)
    
    Returns:
        Image with keypoints drawn
    """
    vis_img = image.copy()
    
    for point in points:
        # Transform point using homography
        transformed_point = transform_point(point, homography)
        
        # Draw circle at point
        cv2.circle(vis_img, transformed_point, radius, color, -1)
    
    return vis_img

def plot_training_progress(losses, save_path=None):
    """
    Plot training loss curve.
    
    Args:
        losses: List of loss values
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()
