"""
Human-readable info for the 38 PlantVillage-style class names produced by the
CNN in 02_disease_detection_cnn.ipynb (e.g. "Tomato___Early_blight").

Only the most common classes are curated with full agronomic detail below;
anything not listed falls back to a sensible generic response built from the
class name itself, so /api/disease/predict never breaks on an unseen class.
Extend this dictionary freely as your review panel asks for more crops.
"""

DISEASE_INFO = {
    "Tomato___Early_blight": {
        "plant": "Tomato", "disease": "Early Blight", "pathogen": "Fungal",
        "symptoms": ["Concentric target-like brown lesions on older leaves",
                     "Yellow chlorotic halo around spots", "Premature leaf drop"],
        "organic": ["Neem oil (5ml/L)", "Trichoderma harzianum soil application"],
        "chemical": ["Mancozeb 75% WP @ 2g/L water"],
    },
    "Tomato___Late_blight": {
        "plant": "Tomato", "disease": "Late Blight", "pathogen": "Fungal (Oomycete)",
        "symptoms": ["Water-soaked dark green/brown lesions", "White fungal growth under leaves in humid weather"],
        "organic": ["Copper oxychloride spray"],
        "chemical": ["Metalaxyl + Mancozeb combination fungicide"],
    },
    "Tomato___healthy": {
        "plant": "Tomato", "disease": "Healthy", "pathogen": "None",
        "symptoms": ["No visible lesions, uniform green color"],
        "organic": [], "chemical": [],
    },
    "Potato___Early_blight": {
        "plant": "Potato", "disease": "Early Blight", "pathogen": "Fungal",
        "symptoms": ["Dark brown concentric ring spots on lower leaves"],
        "organic": ["Bordeaux mixture 1%"], "chemical": ["Mancozeb 75% WP @ 2g/L"],
    },
    "Potato___Late_blight": {
        "plant": "Potato", "disease": "Late Blight", "pathogen": "Fungal (Oomycete)",
        "symptoms": ["Water-soaked lesions turning brown/black rapidly", "White mold on underside in humid conditions"],
        "organic": ["Copper-based fungicide"], "chemical": ["Cymoxanil + Mancozeb"],
    },
    "Potato___healthy": {
        "plant": "Potato", "disease": "Healthy", "pathogen": "None",
        "symptoms": ["No visible lesions"], "organic": [], "chemical": [],
    },
    "Corn_(maize)___Common_rust_": {
        "plant": "Corn (Maize)", "disease": "Common Rust", "pathogen": "Fungal",
        "symptoms": ["Small circular to elongate cinnamon-brown pustules on both leaf surfaces"],
        "organic": ["Resistant hybrid seed selection"], "chemical": ["Propiconazole spray"],
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Corn (Maize)", "disease": "Northern Leaf Blight", "pathogen": "Fungal",
        "symptoms": ["Long cigar-shaped grey-green to tan lesions"],
        "organic": ["Crop rotation, residue management"], "chemical": ["Azoxystrobin + Propiconazole"],
    },
    "Corn_(maize)___healthy": {
        "plant": "Corn (Maize)", "disease": "Healthy", "pathogen": "None",
        "symptoms": ["No visible lesions"], "organic": [], "chemical": [],
    },
    "Apple___Apple_scab": {
        "plant": "Apple", "disease": "Apple Scab", "pathogen": "Fungal",
        "symptoms": ["Olive-green to black velvety spots on leaves and fruit"],
        "organic": ["Sulfur spray"], "chemical": ["Captan or Myclobutanil"],
    },
    "Apple___Black_rot": {
        "plant": "Apple", "disease": "Black Rot", "pathogen": "Fungal",
        "symptoms": ["Purple-bordered leaf spots", "Concentric-ringed fruit rot"],
        "organic": ["Prune and destroy infected wood"], "chemical": ["Captan spray"],
    },
    "Apple___healthy": {
        "plant": "Apple", "disease": "Healthy", "pathogen": "None",
        "symptoms": ["No visible lesions"], "organic": [], "chemical": [],
    },
    "Grape___Black_rot": {
        "plant": "Grape", "disease": "Black Rot", "pathogen": "Fungal",
        "symptoms": ["Circular tan leaf spots with dark border", "Shriveled black mummified berries"],
        "organic": ["Remove mummified berries"], "chemical": ["Mancozeb spray"],
    },
    "Grape___healthy": {
        "plant": "Grape", "disease": "Healthy", "pathogen": "None",
        "symptoms": ["No visible lesions"], "organic": [], "chemical": [],
    },
    "Pepper,_bell___Bacterial_spot": {
        "plant": "Bell Pepper", "disease": "Bacterial Spot", "pathogen": "Bacterial",
        "symptoms": ["Small water-soaked spots turning brown with yellow halo"],
        "organic": ["Copper-based bactericide"], "chemical": ["Copper hydroxide + Mancozeb"],
    },
    "Pepper,_bell___healthy": {
        "plant": "Bell Pepper", "disease": "Healthy", "pathogen": "None",
        "symptoms": ["No visible lesions"], "organic": [], "chemical": [],
    },
}


def get_disease_info(class_name: str) -> dict:
    """Return curated info if available, else a generic fallback built from the class name."""
    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]

    # Fallback: parse "Plant___Disease_name" into readable fields
    parts = class_name.split("___")
    plant = parts[0].replace("_", " ").strip() if parts else "Unknown Plant"
    disease_raw = parts[1] if len(parts) > 1 else "Unknown"
    is_healthy = "healthy" in disease_raw.lower()
    disease = "Healthy" if is_healthy else disease_raw.replace("_", " ").strip()

    return {
        "plant": plant,
        "disease": disease,
        "pathogen": "None" if is_healthy else "See agronomy reference for pathogen type",
        "symptoms": [] if is_healthy else ["Visual symptoms consistent with " + disease],
        "organic": [] if is_healthy else ["Consult local agricultural extension for organic treatment options"],
        "chemical": [] if is_healthy else ["Consult local agricultural extension for approved fungicide/bactericide"],
    }
