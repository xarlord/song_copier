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

/* === Tab number badges (keyboard shortcut indicators) === */
.tab-nav button { position: relative; }
.tab-nav button::after {
    content: attr(data-tab-idx);
    display: none; /* toggled via JS below */
}
.tab-badge {
    position: absolute;
    top: 2px;
    right: 4px;
    font-size: 0.65rem !important;
    background: rgba(83, 52, 131, 0.45);
    color: rgba(255,255,255,0.5);
    border-radius: 4px;
    padding: 1px 4px;
    line-height: 1;
    pointer-events: none;
}

/* === Global Audio Player Footer === */
.ag-audio-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 9999;
    background: linear-gradient(180deg, #12122a 0%, #0a0a1e 100%);
    border-top: 1px solid rgba(83, 52, 131, 0.35);
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 10px 24px;
    height: 56px;
    box-sizing: border-box;
    backdrop-filter: blur(12px);
}
.ag-footer-track-info {
    display: flex;
    flex-direction: column;
    min-width: 140px;
    max-width: 220px;
}
.ag-footer-track-name {
    color: #d0d0f0;
    font-size: 0.85rem;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.ag-footer-track-time {
    color: #6a6a8a;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
}
.ag-footer-controls {
    display: flex;
    gap: 6px;
    align-items: center;
}
.ag-footer-btn {
    background: rgba(83, 52, 131, 0.25);
    border: 1px solid rgba(83, 52, 131, 0.35);
    color: #c8c8e8;
    border-radius: 50%;
    width: 34px;
    height: 34px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 0.9rem;
    transition: all 0.15s ease;
    padding: 0;
    line-height: 1;
}
.ag-footer-btn:hover {
    background: rgba(83, 52, 131, 0.5);
    color: #fff;
}
.ag-footer-btn-play {
    width: 40px;
    height: 40px;
    font-size: 1rem;
    background: linear-gradient(135deg, #533483, #0f3460);
    border: none;
    box-shadow: 0 2px 8px rgba(83, 52, 131, 0.35);
}
.ag-footer-btn-play:hover {
    box-shadow: 0 4px 14px rgba(83, 52, 131, 0.55);
    transform: scale(1.05);
}
.ag-footer-progress-wrap {
    flex: 1;
    min-width: 80px;
}
.ag-footer-progress-bar {
    height: 6px;
    background: rgba(255,255,255,0.08);
    border-radius: 3px;
    cursor: pointer;
    position: relative;
    overflow: hidden;
}
.ag-footer-progress-bar:hover {
    height: 10px;
}
.ag-footer-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #533483, #7b5ea7);
    border-radius: 3px;
    transition: width 0.1s linear;
}
.ag-footer-volume-wrap {
    display: flex;
    align-items: center;
    gap: 6px;
    min-width: 110px;
}
.ag-footer-volume-icon {
    font-size: 1rem;
    cursor: default;
}
.ag-footer-volume-slider {
    -webkit-appearance: none;
    appearance: none;
    width: 80px;
    height: 4px;
    background: rgba(255,255,255,0.1);
    border-radius: 2px;
    outline: none;
}
.ag-footer-volume-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #7b5ea7;
    cursor: pointer;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3);
}
.ag-footer-volume-slider::-moz-range-thumb {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #7b5ea7;
    cursor: pointer;
    border: none;
}

/* Play-in-footer button added to Gradio audio players */
.ag-play-in-footer {
    position: absolute;
    top: 4px;
    right: 4px;
    background: rgba(83, 52, 131, 0.6);
    border: none;
    border-radius: 6px;
    padding: 2px 6px;
    font-size: 0.9rem;
    cursor: pointer;
    opacity: 0;
    transition: opacity 0.2s ease;
    z-index: 10;
}
.ag-play-in-footer:hover {
    background: rgba(83, 52, 131, 0.9);
}
audio:hover ~ .ag-play-in-footer,
.ag-play-in-footer:hover {
    opacity: 1;
}

/* Add bottom padding to the main container so content isn't hidden behind the footer */
.gradio-container {
    padding-bottom: 70px !important;
}

/* === Better Scrollbar === */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: rgba(10, 10, 26, 0.5);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb {
    background: rgba(83, 52, 131, 0.4);
    border-radius: 4px;
    transition: background 0.2s ease;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(83, 52, 131, 0.65);
}
* {
    scrollbar-width: thin;
    scrollbar-color: rgba(83, 52, 131, 0.4) rgba(10, 10, 26, 0.5);
}

/* === Loading Spinner Animation === */
@keyframes ag-spin {
    to { transform: rotate(360deg); }
}
.ag-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 2px solid rgba(83, 52, 131, 0.2);
    border-top-color: #7b5ea7;
    border-radius: 50%;
    animation: ag-spin 0.7s linear infinite;
    vertical-align: middle;
}
button:disabled {
    position: relative;
}
button:disabled::after {
    content: '';
    position: absolute;
    right: 10px;
    top: 50%;
    margin-top: -8px;
    width: 16px;
    height: 16px;
    border: 2px solid rgba(255,255,255,0.15);
    border-top-color: rgba(255,255,255,0.6);
    border-radius: 50%;
    animation: ag-spin 0.7s linear infinite;
}

/* === Toast Notification === */
.ag-toast {
    position: fixed;
    bottom: 70px;
    left: 50%;
    transform: translateX(-50%) translateY(20px);
    background: linear-gradient(135deg, #533483, #0f3460);
    color: #fff;
    padding: 10px 24px;
    border-radius: 10px;
    font-size: 0.9rem;
    font-weight: 500;
    box-shadow: 0 6px 24px rgba(0,0,0,0.4);
    opacity: 0;
    pointer-events: none;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    z-index: 10000;
    white-space: nowrap;
}
.ag-toast-visible {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
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
    .ag-audio-footer {
        padding: 8px 12px;
        gap: 8px;
        height: 50px;
    }
    .ag-footer-track-info {
        min-width: 80px;
        max-width: 120px;
    }
    .ag-footer-track-name { font-size: 0.75rem; }
    .ag-footer-volume-wrap { display: none; }
    .tab-nav { overflow-x: auto !important; flex-wrap: nowrap !important; }
    .tab-nav button { flex-shrink: 0; font-size: 0.82rem !important; padding: 8px 12px !important; }
}
@media (max-width: 480px) {
    .ag-footer-track-info { display: none; }
    .ag-footer-progress-wrap { min-width: 50px; }
}
"""
