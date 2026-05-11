"""Custom CSS for Audio Generator Studio — dark professional theme."""

CUSTOM_CSS = """
/* === Base === */
footer { display: none !important; }
.gradio-container {
    max-width: 1280px !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

/* === Header === */
.app-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 28px 36px !important;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.06);
}
.app-header h1 {
    color: #e0e0ff !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin-bottom: 6px !important;
}
.app-header p {
    color: #8888aa !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
}
.app-header .device-info {
    color: #53a8b6 !important;
    font-size: 0.85rem !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}

/* === Section cards === */
.section-card {
    background: rgba(30, 30, 50, 0.5);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 20px 24px !important;
    margin-bottom: 16px;
}

/* === Tab styling === */
.tabs {
    border: none !important;
}
.tab-nav {
    background: rgba(20, 20, 40, 0.6) !important;
    border-radius: 12px !important;
    padding: 6px !important;
    gap: 4px !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
}
.tab-nav button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
}
.tab-nav button.selected {
    background: linear-gradient(135deg, #533483, #0f3460) !important;
    color: #fff !important;
    box-shadow: 0 2px 8px rgba(83, 52, 131, 0.3) !important;
}

/* === Section headings === */
.section-card h3, h3 {
    color: #b8b8d0 !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding-bottom: 10px !important;
    margin-bottom: 16px !important;
}

/* === Accordion styling === */
.accordions {
    border: none !important;
}
.accordion {
    background: rgba(25, 25, 45, 0.4) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
}
.accordion .label-wrap {
    color: #c0c0d8 !important;
    font-weight: 500 !important;
}

/* === Buttons === */
.primary-btn, button.primary {
    background: linear-gradient(135deg, #533483, #0f3460) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(83, 52, 131, 0.25) !important;
}
.primary-btn:hover, button.primary:hover {
    box-shadow: 0 4px 16px rgba(83, 52, 131, 0.4) !important;
    transform: translateY(-1px) !important;
}

button.secondary, button:not(.primary) {
    border-radius: 8px !important;
    font-weight: 500 !important;
}

/* === Slider styling === */
input[type="range"] {
    accent-color: #533483 !important;
}

/* === Audio components === */
.audio-container {
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* === Status textbox === */
.status-box textarea {
    background: rgba(20, 20, 40, 0.6) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: #53a8b6 !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.85rem !important;
}

/* === Dropdown === */
.dropdown-container {
    border-radius: 8px !important;
}

/* === Stem color indicators === */
.stem-vocals { border-left: 3px solid #e74c3c !important; padding-left: 10px; }
.stem-drums { border-left: 3px solid #f39c12 !important; padding-left: 10px; }
.stem-bass { border-left: 3px solid #2ecc71 !important; padding-left: 10px; }
.stem-guitar { border-left: 3px solid #9b59b6 !important; padding-left: 10px; }
.stem-piano { border-left: 3px solid #3498db !important; padding-left: 10px; }
.stem-other { border-left: 3px solid #95a5a6 !important; padding-left: 10px; }

/* === Preset bar === */
.preset-bar {
    background: rgba(25, 25, 50, 0.5);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 14px 20px !important;
}

/* === Quick-action buttons row === */
.actions-row {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
}

/* === Upload zone === */
.upload-zone {
    border: 2px dashed rgba(83, 52, 131, 0.4) !important;
    border-radius: 12px !important;
    min-height: 120px !important;
    transition: border-color 0.2s ease !important;
}
.upload-zone:hover {
    border-color: rgba(83, 52, 131, 0.7) !important;
}

/* === Responsive === */
@media (max-width: 768px) {
    .app-header {
        padding: 18px 20px !important;
        border-radius: 12px;
    }
    .app-header h1 {
        font-size: 1.4rem !important;
    }
    .section-card {
        padding: 14px 16px !important;
    }
}
"""
