# Flask Web Application Architecture

## Overview

This web application provides a simple academic interface for uploading a waste image and receiving:

- predicted waste class
- confidence score
- RDF suitability prediction

It uses the existing multimodal inference layer and does not add authentication or deployment logic.

## Request Flow

1. The user uploads a waste image through the homepage form.
2. The Flask app validates the uploaded file.
3. The image is passed to the multimodal inference pipeline.
4. The pipeline returns:
   - waste class prediction
   - classifier confidence
   - RDF suitability label
5. The result is rendered on the same page with a clean academic layout.

## Components

| Component | File | Purpose |
|---|---|---|
| Flask app | `app.py` | Handles routes, validation, and response rendering |
| Multimodal pipeline | `src/multimodal_inference.py` | Chains image classification to RDF prediction |
| HTML template | `templates/index.html` | Upload form and result display |
| Styling | `static/style.css` | Academic visual presentation |

## Model Dependencies

- Image classifier: `models/cnn_baseline_best.h5`
- RDF predictor: `models/rdf_random_forest_pipeline.joblib`

## Design Notes

- The app keeps inference logic outside the route handler.
- The pipeline is cached so the models are loaded once per process.
- Uploaded files are validated before inference and handled as temporary files.
- The interface is intentionally minimal to keep the dissertation prototype focused.

## Limitations

- No authentication.
- No deployment configuration.
- No batch upload flow.
- No direct fusion of image and tabular features beyond the class-to-feature mapping.
