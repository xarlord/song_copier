"""Project Library tab — browse, search, and reload saved stem separation sessions."""

import gradio as gr

from services.project_manager import (
    list_projects,
    load_project,
    delete_project,
    search_projects,
)


# Column headers for the project list dataframe
_DF_HEADERS = ["Name", "Original", "Date", "BPM", "Key", "Stems"]


def _projects_to_rows(projects: list[dict]) -> list[list]:
    """Convert project dicts to dataframe rows."""
    rows = []
    for p in projects:
        bpm_str = f"{p['bpm']:.0f}" if p.get("bpm") else "—"
        rows.append([
            p.get("name", ""),
            p.get("original_filename", ""),
            p.get("date", "")[:10] if p.get("date") else "",  # date only
            bpm_str,
            p.get("key", "") or "—",
            str(p.get("stem_count", 0)),
        ])
    return rows


def create_tab() -> gr.Column:
    """Create the Project Library tab UI.

    Returns:
        gr.Column containing the full tab layout.
    """
    with gr.Column() as tab:
        gr.Markdown("## 📂 Project Library")
        gr.Markdown(
            "Browse, search, and reload your saved stem separation sessions. "
            "Click any row to view project details and play individual stems."
        )

        # ── Search & Refresh ─────────────────────────────────────
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### Search & Filter")
            with gr.Row():
                search_box = gr.Textbox(
                    label="Search",
                    placeholder="Type name, filename, or 'bpm:120-140'",
                    scale=4,
                )
                search_btn = gr.Button("🔍 Search", size="sm", scale=1)
                refresh_btn = gr.Button("🔄 Refresh", size="sm", scale=1)
                delete_btn = gr.Button("🗑️ Delete Selected", size="sm", scale=1)

        # ── Project List ──────────────────────────────────────────
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### Saved Projects")
            project_df = gr.Dataframe(
                headers=_DF_HEADERS,
                datatype=["str", "str", "str", "str", "str", "str"],
                label="Projects",
                interactive=False,
                wrap=True,
                column_widths=[180, 200, 100, 70, 80, 60],
            )

        status_msg = gr.Textbox(
            label="Status",
            interactive=False,
            lines=2,
            elem_classes=["status-box"],
        )

        # ── Project Detail (shown when a row is selected) ────────
        detail_section = gr.Column(visible=False)

        with detail_section:
            with gr.Column(elem_classes=["section-card"]):
                gr.Markdown("### Project Details")
                detail_name = gr.Textbox(label="Project Name", interactive=False)
                detail_original = gr.Textbox(label="Original File", interactive=False)
                detail_date = gr.Textbox(label="Date", interactive=False)
                with gr.Row():
                    detail_bpm = gr.Textbox(label="BPM", interactive=False, scale=1)
                    detail_key = gr.Textbox(label="Key", interactive=False, scale=1)
                    detail_duration = gr.Textbox(label="Duration (s)", interactive=False, scale=1)
                    detail_model = gr.Textbox(label="Model", interactive=False, scale=1)
                detail_stem_status = gr.Textbox(
                    label="Stem File Status",
                    interactive=False,
                    lines=2,
                )

            # ── Stem Audio Players (dynamically populated) ────────
            with gr.Column(elem_classes=["section-card"]):
                gr.Markdown("### Stems")
                stem_players = {}
                stem_names = ["vocals", "drums", "bass", "guitar", "piano", "other"]
                with gr.Row():
                    stem_players["vocals"] = gr.Audio(label="🎤 Vocals", type="filepath")
                    stem_players["drums"] = gr.Audio(label="🥁 Drums", type="filepath")
                with gr.Row():
                    stem_players["bass"] = gr.Audio(label="🎸 Bass", type="filepath")
                    stem_players["guitar"] = gr.Audio(label="🎶 Guitar", type="filepath")
                with gr.Row():
                    stem_players["piano"] = gr.Audio(label="🎹 Piano", type="filepath")
                    stem_players["other"] = gr.Audio(label="🎻 Other", type="filepath")

            # ── Reload action ─────────────────────────────────────
            with gr.Column(elem_classes=["section-card"]):
                gr.Markdown("### Actions")
                reload_btn = gr.Button(
                    "🔁 Reload into Stem Separator",
                    variant="primary",
                    size="lg",
                )
                reload_status = gr.Textbox(visible=False, show_label=False)

        # ── State ─────────────────────────────────────────────────
        selected_project_name = gr.State(None)

        # ── Callbacks ─────────────────────────────────────────────

        def do_refresh():
            """Reload project list from disk."""
            projects = list_projects()
            rows = _projects_to_rows(projects)
            count = len(rows)
            msg = f"Loaded {count} project{'s' if count != 1 else ''}."
            return (
                gr.Dataframe(value=rows),
                msg,
                gr.Column(visible=False),   # hide detail on refresh
                None,                        # clear selected name
            )

        def do_search(query: str):
            """Search projects by query string."""
            results = search_projects(query or "")
            rows = _projects_to_rows(results)
            count = len(rows)
            if not query or not query.strip():
                msg = f"Showing all {count} project{'s' if count != 1 else ''}."
            else:
                msg = f"Found {count} project{'s' if count != 1 else ''} matching '{query.strip()}'."
            return (
                gr.Dataframe(value=rows),
                msg,
                gr.Column(visible=False),
                None,
            )

        def do_select_row(evt: gr.SelectData):
            """Handle row click in the project dataframe."""
            if evt.index is None or evt.index[0] is None:
                return (
                    gr.Column(visible=False),
                    "", "", "", "", "", "", "", "",
                    *[None] * 6,
                    None,
                    "",
                )

            row_idx = evt.index[0]
            projects = list_projects()
            if row_idx >= len(projects):
                return (
                    gr.Column(visible=False),
                    "", "", "", "", "", "", "", "",
                    *[None] * 6,
                    None,
                    "",
                )

            proj = projects[row_idx]
            name = proj.get("name", "")

            # Load full project with stem verification
            data = load_project(name)
            if data is None:
                return (
                    gr.Column(visible=False),
                    "Project not found.", "", "", "", "", "", "", "",
                    *[None] * 6,
                    None,
                    "",
                )

            # Build stem status text
            stem_status = data.get("stem_status", {})
            status_parts = []
            for sname, exists in stem_status.items():
                icon = "✅" if exists else "❌"
                status_parts.append(f"{icon} {sname}")
            if data.get("missing_stems"):
                status_parts.append(
                    f"\n⚠️ {len(data['missing_stems'])} stem file(s) missing"
                )
            stem_status_text = "\n".join(status_parts) if status_parts else "No stems recorded."

            # Duration
            dur = data.get("duration")
            dur_str = f"{dur:.1f}" if dur else "—"

            # BPM
            bpm = data.get("bpm")
            bpm_str = f"{bpm:.0f}" if bpm else "—"

            # Stem audio paths (None if missing)
            stem_audio = []
            verified = data.get("verified_stems", {})
            for sname in stem_names:
                stem_audio.append(verified.get(sname, None))

            return (
                gr.Column(visible=True),
                data.get("name", ""),
                data.get("original_filename", ""),
                data.get("date", "")[:10] if data.get("date") else "",
                bpm_str,
                data.get("key", "") or "—",
                dur_str,
                data.get("model", "") or "—",
                stem_status_text,
                *stem_audio,
                name,       # selected_project_name state
                f"Loaded project: {name}",
            )

        def do_delete(selected_name: str):
            """Delete the selected project."""
            if not selected_name:
                return (
                    gr.Dataframe(value=_projects_to_rows(list_projects())),
                    "⚠️ No project selected. Click a row first.",
                    gr.Column(visible=False),
                    None,
                )
            success = delete_project(selected_name)
            projects = list_projects()
            rows = _projects_to_rows(projects)
            if success:
                msg = f"✅ Deleted project: {selected_name}"
            else:
                msg = f"❌ Could not delete project: {selected_name}"
            return (
                gr.Dataframe(value=rows),
                msg,
                gr.Column(visible=False),
                None,
            )

        def do_reload_to_separator(selected_name: str):
            """Provide instructions for reloading project into Stem Separator."""
            if not selected_name:
                return gr.update(
                    value="⚠️ No project selected.",
                    visible=True,
                )
            data = load_project(selected_name)
            if data is None:
                return gr.update(
                    value=f"❌ Could not load project: {selected_name}",
                    visible=True,
                )

            original = data.get("original_file", "N/A")
            stems_dir = data.get("stems_dir", "N/A")
            missing = data.get("missing_stems", [])
            stems_list = data.get("stems_list", [])

            if missing:
                warn = f"\n⚠️ Missing stems: {', '.join(missing)}"
            else:
                warn = ""

            info = (
                f"Project: {selected_name}\n"
                f"Original: {original}\n"
                f"Stems directory: {stems_dir}\n"
                f"Stems: {', '.join(stems_list)}"
                f"{warn}\n\n"
                f"📋 To reload: Use the Stem Separator tab and upload the original file, "
                f"or browse to the stems directory above."
            )
            return gr.update(value=info, visible=True)

        # ── Wire callbacks ────────────────────────────────────────

        refresh_btn.click(
            fn=do_refresh,
            outputs=[project_df, status_msg, detail_section, selected_project_name],
        )

        search_btn.click(
            fn=do_search,
            inputs=[search_box],
            outputs=[project_df, status_msg, detail_section, selected_project_name],
        )

        # Also search on Enter key in search box
        search_box.submit(
            fn=do_search,
            inputs=[search_box],
            outputs=[project_df, status_msg, detail_section, selected_project_name],
        )

        project_df.select(
            fn=do_select_row,
            outputs=[
                detail_section,
                detail_name,
                detail_original,
                detail_date,
                detail_bpm,
                detail_key,
                detail_duration,
                detail_model,
                detail_stem_status,
                *[stem_players[s] for s in stem_names],
                selected_project_name,
                status_msg,
            ],
        )

        delete_btn.click(
            fn=do_delete,
            inputs=[selected_project_name],
            outputs=[project_df, status_msg, detail_section, selected_project_name],
        )

        reload_btn.click(
            fn=do_reload_to_separator,
            inputs=[selected_project_name],
            outputs=[reload_status],
        )

        # Populate initial data
        _initial_projects = list_projects()
        _initial_rows = _projects_to_rows(_initial_projects)
        project_df.value = _initial_rows

    return tab
