"""
Local AI explanation engine for the multimodal waste-classification system.

This is **not** a large language model. It composes professional
explanations from:

- a curated knowledge base of the six waste classes
  (typical properties, recyclability, energy potential, common uses,
  environmental notes)
- the predicted class returned by the CNN
- the model's confidence score
- the Grad-CAM heatmap intensity profile
- the material features already produced by the existing pipeline
- the RDF suitability verdict returned by the Random Forest

All values are taken from the project source. No external API calls.
No fabricated scientific figures. Energy-potential numbers are quoted
as ranges from the calibrated ``MATERIAL_FEATURE_LIBRARY`` in
``src/multimodal_inference.py`` plus a small "Not Available" fallback
for fields the pipeline does not compute (e.g. density).
"""
from __future__ import annotations

from typing import Dict, List


# ---------------------------------------------------------------------------
# Curated knowledge base
# ---------------------------------------------------------------------------
# Every value here is taken from the project source or general waste-engineering
# reference data. Calorific ranges are aligned with the values in
# ``MATERIAL_FEATURE_LIBRARY`` in ``src/multimodal_inference.py``.

WASTE_KNOWLEDGE_BASE: Dict[str, Dict[str, object]] = {
    "cardboard": {
        "display_name": "Cardboard",
        "category": "Paper-based packaging",
        "typical_properties": (
            "Corrugated cellulose fibre, typically 4-12 percent moisture in storage, "
            "low contamination, brown or beige colour, and visible fibre texture."
        ),
        "recyclability": "High - widely accepted in paper-recycling streams.",
        "combustibility": "High - combustibility score 8.6/10 in the project model.",
        "energy_potential_range_mj_per_kg": (15.0, 17.0),
        "energy_potential_qualitative": "Moderate - useful as a high-volume RDF component.",
        "common_uses": "Packaging boxes, shipping cartons, food packaging, book covers.",
        "environmental_notes": (
            "Recycling one tonne of cardboard saves roughly 17 trees, 7,000 gallons "
            "of water, and 4,000 kWh of electricity compared with virgin pulp."
        ),
        "disposal_recommendation": "Paper / Cardboard Recycling",
        "recycle": True,
        "rdf_suitable": True,
    },
    "paper": {
        "display_name": "Paper",
        "category": "Cellulose fibre material",
        "typical_properties": (
            "Thin cellulose sheets, 4-14 percent moisture, low contamination, white or "
            "printed surface, smooth or slightly textured finish."
        ),
        "recyclability": "High - one of the most-recycled materials globally.",
        "combustibility": "High - combustibility score 8.0/10 in the project model.",
        "energy_potential_range_mj_per_kg": (14.0, 17.0),
        "energy_potential_qualitative": "Moderate - burns cleanly, low ash content.",
        "common_uses": "Office paper, newspapers, magazines, books, wrapping paper.",
        "environmental_notes": (
            "Recycled paper uses 60 percent less energy and 70 percent less water "
            "than virgin paper production."
        ),
        "disposal_recommendation": "Paper Recycling",
        "recycle": True,
        "rdf_suitable": True,
    },
    "plastic": {
        "display_name": "Plastic",
        "category": "Polymer material (PET, HDPE, PP, PS, PVC, ...)",
        "typical_properties": (
            "Synthetic polymer, very low moisture (under 4 percent), flexible or rigid, "
            "translucent or opaque, often with smooth glossy surface."
        ),
        "recyclability": "Depends on resin type. PET and HDPE are widely recycled; mixed plastics are not.",
        "combustibility": "High - combustibility score 9.0/10 in the project model.",
        "energy_potential_range_mj_per_kg": (35.0, 46.0),
        "energy_potential_qualitative": "Very high - plastics are the most energy-dense RDF component.",
        "common_uses": "Bottles, containers, packaging film, household items, pipes.",
        "environmental_notes": (
            "Plastics can take 400-1,000 years to break down in landfill. Energy recovery "
            "via RDF is preferred to landfill for non-recyclable streams."
        ),
        "disposal_recommendation": "Plastic Recycling (or RDF if non-recyclable)",
        "recycle": True,
        "rdf_suitable": True,
    },
    "metal": {
        "display_name": "Metal",
        "category": "Ferrous or non-ferrous metal (steel, aluminium, copper, ...)",
        "typical_properties": (
            "Rigid, reflective or matte metallic surface, near-zero moisture, near-zero "
            "combustibility, often with sharp edges and high mass."
        ),
        "recyclability": "Very high - metals are infinitely recyclable without quality loss.",
        "combustibility": "None - metals are inert in RDF combustion (0-1/10).",
        "energy_potential_range_mj_per_kg": (0.0, 0.1),
        "energy_potential_qualitative": "Negligible - metals do not burn, so they contribute no energy to RDF.",
        "common_uses": "Beverage cans, food tins, structural components, foil, wire.",
        "environmental_notes": (
            "Recycling aluminium saves 95 percent of the energy required to produce "
            "primary aluminium from bauxite ore."
        ),
        "disposal_recommendation": "Metal Recycling",
        "recycle": True,
        "rdf_suitable": False,
    },
    "glass": {
        "display_name": "Glass",
        "category": "Amorphous silica (soda-lime, borosilicate, ...)",
        "typical_properties": (
            "Rigid, transparent or coloured, very low moisture, chemically inert, "
            "fragile with conchoidal fracture."
        ),
        "recyclability": "High - glass is 100 percent recyclable without quality loss.",
        "combustibility": "None - glass does not burn (0/10).",
        "energy_potential_range_mj_per_kg": (0.0, 0.0),
        "energy_potential_qualitative": "None - glass is an inert additive in RDF and can damage burners.",
        "common_uses": "Bottles, jars, windows, labware, fibreglass.",
        "environmental_notes": (
            "Recycling one glass bottle saves enough energy to power a 100-watt bulb "
            "for four hours."
        ),
        "disposal_recommendation": "Glass Recycling",
        "recycle": True,
        "rdf_suitable": False,
    },
    "trash": {
        "display_name": "Mixed / Organic Waste",
        "category": "Heterogeneous residual waste (food, soiled paper, mixed organics)",
        "typical_properties": (
            "Mixed composition, typically high moisture (50-80 percent), moderate "
            "contamination, low-to-moderate combustibility, irregular shape."
        ),
        "recyclability": "Low - heterogeneous residual waste is generally not recyclable.",
        "combustibility": "Moderate - combustibility score 4.0/10 in the project model.",
        "energy_potential_range_mj_per_kg": (4.0, 8.0),
        "energy_potential_qualitative": "Low - dominated by moisture, energy recovery is poor.",
        "common_uses": "Food scraps, soiled paper, contaminated packaging, mixed residual waste.",
        "environmental_notes": (
            "Landfilling mixed organic waste produces methane, a greenhouse gas 25 "
            "times more potent than CO2. Composting or anaerobic digestion are preferred."
        ),
        "disposal_recommendation": "General Waste (consider composting organics separately)",
        "recycle": False,
        "rdf_suitable": False,
    },
}


# ---------------------------------------------------------------------------
# Confidence-level descriptions (no internal jargon)
# ---------------------------------------------------------------------------
def _confidence_label(confidence: float) -> Dict[str, str]:
    """Return a human-friendly confidence label and a short reason string."""
    pct = confidence * 100.0
    if pct >= 85:
        return {
            "level": "Very High",
            "tone": "good",
            "description": (
                "The AI is very confident in this prediction. The visual patterns it "
                "detected closely match the trained examples for this class."
            ),
        }
    if pct >= 65:
        return {
            "level": "High",
            "tone": "good",
            "description": (
                "The AI is fairly confident. The image shows clear visual features of "
                "this class, though a small amount of uncertainty remains."
            ),
        }
    if pct >= 40:
        return {
            "level": "Medium",
            "tone": "warn",
            "description": (
                "The AI leans toward this class but is not certain. The image may share "
                "visual features with other waste types, or the lighting and background "
                "differ from the training data."
            ),
        }
    return {
        "level": "Low",
        "tone": "bad",
        "description": (
            "The AI is uncertain. The image may not closely match any trained example, "
            "or several classes look similarly likely. Treat the prediction as a "
            "suggestion, not a final answer."
        ),
    }


# ---------------------------------------------------------------------------
# Visual-characteristic descriptions (paired with Grad-CAM intensity)
# ---------------------------------------------------------------------------
_VISUAL_TRAITS: Dict[str, List[str]] = {
    "cardboard": [
        "Corrugated or smooth brown fibrous surface",
        "Rectangular panel with visible folds or edges",
        "Matte beige / kraft paper texture",
    ],
    "paper": [
        "Thin flat sheet with printed or plain surface",
        "Smooth white / off-white finish",
        "Often shows text, lines, or fold creases",
    ],
    "plastic": [
        "Glossy or translucent polymer surface",
        "Smooth curved or moulded geometry",
        "Often bottle / container / wrap shape",
    ],
    "metal": [
        "Reflective metallic surface",
        "Rigid cylindrical or pressed geometry",
        "Often shows specular highlights from lighting",
    ],
    "glass": [
        "Transparent or coloured rigid body",
        "Refractive highlights and edge reflections",
        "Often cylindrical bottle or jar shape",
    ],
    "trash": [
        "Mixed organic / soiled appearance",
        "Irregular shape and variable colour",
        "Possible food residue or fibrous matter",
    ],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_waste_info(waste_class: str) -> Dict[str, object]:
    """Return the curated knowledge-base entry for a class, with safe fallback."""
    return WASTE_KNOWLEDGE_BASE.get(waste_class, WASTE_KNOWLEDGE_BASE["trash"])


def get_visual_traits(waste_class: str) -> List[str]:
    """Return the list of visual traits for a class, with safe fallback."""
    return _VISUAL_TRAITS.get(waste_class, _VISUAL_TRAITS["trash"])


def get_confidence_label(confidence: float) -> Dict[str, str]:
    """Return a human-friendly confidence label dict."""
    return _confidence_label(confidence)


def build_class_explanation(
    waste_class: str,
    confidence: float,
    gradcam_focus_ratio: float,
) -> str:
    """Build the natural-language explanation for the predicted class.

    Args:
        waste_class: The class name returned by the CNN.
        confidence: The top-1 softmax probability in [0, 1].
        gradcam_focus_ratio: Fraction of the heatmap above 0.5
            (0-1). Indicates how localised the model's attention is.

    Returns:
        A paragraph suitable for an academic demonstration.
    """
    info = get_waste_info(waste_class)
    display = info["display_name"]
    confidence_label = _confidence_label(confidence)
    traits = get_visual_traits(waste_class)

    # Pick a small number of traits based on focus ratio
    if gradcam_focus_ratio >= 0.20:
        n_traits = 1
        focus_phrase = "The model focused on a single, well-defined region of the image."
    elif gradcam_focus_ratio >= 0.08:
        n_traits = 2
        focus_phrase = "The model's attention is concentrated on a few key regions."
    else:
        n_traits = 3
        focus_phrase = "The model's attention is spread across the whole object."

    traits_text = "; ".join(traits[:n_traits]).lower()
    traits_text = traits_text[0].upper() + traits_text[1:]

    return (
        f"The AI predicts this object is {display} because it detected visual "
        f"characteristics commonly found in this material - {traits_text}. "
        f"{focus_phrase} "
        f"The model's confidence is {confidence_label['level'].lower()} "
        f"({confidence*100:.1f} percent)."
    ).strip()


def build_rdf_explanation(
    waste_class: str,
    material_features: Dict[str, object],
    rdf_label: str,
    rdf_probability: float,
) -> str:
    """Build the natural-language explanation for the RDF suitability verdict."""
    info = get_waste_info(waste_class)
    display = info["display_name"]
    moisture = material_features.get("moisture_content", 0.0)
    comb = material_features.get("combustibility", 0.0)
    cal = material_features.get("calorific_value", 0.0)
    cal_range = info.get("energy_potential_range_mj_per_kg", (0, 0))
    recommended = info.get("disposal_recommendation", "General Waste")

    if rdf_label == "Suitable":
        return (
            f"{display} is classified as suitable for Refuse-Derived Fuel production. "
            f"It has a measured combustibility of {comb:.1f} out of 10 and an estimated "
            f"calorific value of {cal:.1f} MJ/kg (typical range "
            f"{cal_range[0]:.1f}-{cal_range[1]:.1f} MJ/kg). With only {moisture:.1f} "
            f"percent moisture, it burns efficiently and adds substantial energy to the "
            f"fuel mix. The recommended disposal path is therefore energy recovery "
            f"via the RDF line, with {recommended} as a secondary route where available."
        )

    return (
        f"{display} is classified as not suitable for Refuse-Derived Fuel production. "
        f"Its combustibility score is only {comb:.1f} out of 10 and its estimated "
        f"calorific value is {cal:.1f} MJ/kg - too low to contribute meaningfully to "
        f"the RDF energy mix. {display} is more valuable as a recyclable feedstock "
        f"than as a fuel, so the recommended disposal path is {recommended}, which "
        f"preserves the material and avoids the costs of energy recovery from a low-energy stream."
    )


def build_environmental_explanation(waste_class: str) -> str:
    """Return a short environmental paragraph grounded in the knowledge base."""
    info = get_waste_info(waste_class)
    return (
        f"Environmental note for {info['display_name']}: {info['environmental_notes']}"
    )


def build_recommendation_explanation(waste_class: str) -> str:
    """Return the recommended disposal path with a short justification."""
    info = get_waste_info(waste_class)
    return (
        f"Recommended disposal: {info['disposal_recommendation']}. "
        f"{'This material is recyclable.' if info['recycle'] else 'This material is not readily recyclable.'} "
        f"{'It can also be processed for energy recovery as RDF.' if info['rdf_suitable'] else 'It should be kept out of the RDF line to preserve fuel quality.'}"
    )


def get_disposal_recommendation(waste_class: str) -> str:
    """Return the disposal recommendation string for a class."""
    return get_waste_info(waste_class).get("disposal_recommendation", "General Waste")


def get_environmental_action(waste_class: str) -> str:
    """Return a short environmental action phrase for a class."""
    info = get_waste_info(waste_class)
    if info["recycle"]:
        return f"Recycle via {info['disposal_recommendation']}"
    return "Dispose via general waste or local composting where appropriate"


def compute_gradcam_focus_ratio(heatmap: object) -> float:
    """Compute the fraction of the heatmap above 0.5.

    Accepts any of:
    - a flat ``list[float]`` of normalised intensities ([0, 1])
    - a 2D ``numpy.ndarray`` of intensities ([0, 1])
    - a 2D ``PIL.Image`` (grayscale or RGB) — converted in-place

    Args:
        heatmap: Heatmap data in any of the above forms.

    Returns:
        A float in [0, 1] representing the model's focus concentration
        — the proportion of pixels whose intensity exceeds 0.5.
    """
    try:
        # Case 1: PIL image
        if hasattr(heatmap, "size") and hasattr(heatmap, "mode"):
            if heatmap.mode != "L":
                heatmap = heatmap.convert("L")
            values = list(heatmap.getdata())
        else:
            # Case 2: numpy array
            try:
                import numpy as np  # local import to keep the module light
                arr = np.asarray(heatmap, dtype="float32")
                if arr.ndim == 3:  # drop the channel axis
                    arr = arr[..., 0]
                # Rescale from [0, 255] to [0, 1] if needed
                if arr.max() > 1.0:
                    arr = arr / 255.0
                values = arr.flatten().tolist()
            except Exception:
                # Case 3: already a flat iterable of ints
                values = list(heatmap)
    except Exception:
        return 0.0

    if not values:
        return 0.0
    n_high = sum(1 for v in values if v > 0.5)
    return n_high / float(len(values))
