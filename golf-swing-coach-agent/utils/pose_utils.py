import numpy as np

def calculate_angle(a, b, c):
    """Calculate angle (in degrees) between three 3D points"""
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)


def detect_swing_phases(frames_data):
    """Placeholder - improve later with better logic"""
    if len(frames_data) < 10:
        return {"error": "Not enough frames"}
    return {
        "address": "detected",
        "top_of_backswing": "detected",
        "impact": "detected",
        "follow_through": "detected",
        "note": "Phase detection will be enhanced in next version"
    }