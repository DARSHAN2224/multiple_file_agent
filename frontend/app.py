import streamlit as st
import requests
import uuid
import datetime

API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="Multi-Document Agent", layout="wide")
st.title("Enterprise Multi-Document Analysis Agent")

# ── Session Initialisation ──────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── 1. Session Management ──────────────────────────────────────────────
    st.markdown("### 🔑 Sessions")

    # Styled current session ID badge
    short_id = st.session_state.session_id[:8]
    st.markdown(
        f"""
        <div style="background:#1e1e2e;border:1px solid #3a3a5c;border-radius:8px;
                    padding:8px 12px;margin-bottom:8px;font-family:monospace;font-size:12px;color:#a0a0c0;">
          <span style="color:#6c6cff;font-weight:600;">ACTIVE</span>&nbsp;&nbsp;
          <span style="color:#e0e0ff;">{st.session_state.session_id}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_new, col_manual = st.columns([1, 1])
    with col_new:
        if st.button("＋ New", key="new_session_btn", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
    with col_manual:
        show_manual = st.toggle("ID input", key="show_manual_toggle")

    if show_manual:
        manual_id = st.text_input("Paste session ID:", key="manual_session_input",
                                  label_visibility="collapsed",
                                  placeholder="Paste session UUID…")
        if st.button("Apply ID", key="apply_manual_session"):
            if manual_id.strip():
                st.session_state.session_id = manual_id.strip()
                st.rerun()

    # Fetching saved sessions is no longer needed in the sidebar as they are in the 'Session History' tab.

    st.divider()

    # ── 2. Document Upload ─────────────────────────────────────────────────
    st.header("📄 Document Management")
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, DOCX, TXT, CSV, Excel)",
        type=["pdf", "docx", "txt", "csv", "xlsx", "xls"],
        accept_multiple_files=True
    )

    if st.button("Index Documents"):
        if uploaded_files:
            with st.spinner("Processing & Indexing..."):
                files = [("files", file) for file in uploaded_files]
                data = {"session_id": st.session_state.session_id}
                try:
                    response = requests.post(f"{API_URL}/documents/upload", files=files, data=data)
                    if response.status_code == 200:
                        st.success(f"✅ Successfully processed {response.json().get('total_chunks')} internal chunks.")
                        st.rerun()
                    else:
                        st.error(f"Error: {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")
        else:
            st.warning("Please upload a file first.")

    st.divider()

    # ── 3. Uploaded Files List ─────────────────────────────────────────────
    st.subheader("Uploaded Files")
    try:
        res = requests.get(f"{API_URL}/documents/list/{st.session_state.session_id}", timeout=3)
        if res.status_code == 200:
            docs = res.json().get("documents", [])
            if docs:
                for d in docs:
                    st.text(f"📄 {d}")
            else:
                st.text("No documents indexed yet.")
    except Exception:
        st.text("Could not reach backend.")

# ── Main Tabs ────────────────────────────────────────────────────────────────
tab_qa, tab_compare, tab_history = st.tabs(["Direct Query", "Common Themes & Compare", "🕘 Session History"])

with tab_qa:
    st.header("Query Documents")

    # File scope selector
    try:
        file_list_res = requests.get(f"{API_URL}/documents/list/{st.session_state.session_id}", timeout=3)
        indexed_files = file_list_res.json().get("documents", []) if file_list_res.status_code == 200 else []
    except Exception:
        indexed_files = []

    if indexed_files:
        selected_files = st.multiselect(
            "🗂️ Search in files (leave empty to search all):",
            options=indexed_files,
            default=[],
            placeholder="All files selected by default"
        )
    else:
        selected_files = []
        st.info("Upload and index documents first, or restore a previous session from the sidebar.")

    query = st.text_input("Ask a question across the selected documents:")

    if st.button("Search"):
        if not query:
            st.warning("Please enter a question.")
        else:
            scope_label = ", ".join(selected_files) if selected_files else "all files"
            with st.spinner(f"Searching in {scope_label} (Hybrid Search + Local LLM)..."):
                try:
                    payload = {
                        "session_id": st.session_state.session_id,
                        "query": query,
                        "filter_files": selected_files if selected_files else None
                    }
                    response = requests.post(f"{API_URL}/query", json=payload)

                    if response.status_code == 200:
                        data = response.json()
                        st.markdown("### Answer")
                        st.write(data["answer"])

                        st.markdown(f"**Retrieval Confidence Final Score:** `{data['confidence']:.2f}`")

                        if data["sources"]:
                            st.markdown("### Source Attribution")
                            for idx, s in enumerate(data["sources"]):
                                with st.expander(f"Reference {idx+1}: {s['source_file']} (Page {s.get('page_number', 'N/A')})"):
                                    st.markdown(f"**Section:** {s['section_title']}")
                                    st.markdown(f"*{s['text']}*")
                    else:
                        st.error(f"Backend Error: {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

with tab_compare:
    st.header("Document Comparison Matrix")
    st.write("Automatically cluster contents to find differences and commonalities.")

    if st.button("Generate Matrix"):
        with st.spinner("Clustering topics and summarizing differences..."):
            try:
                payload = {"session_id": st.session_state.session_id, "query": "compare topics"}
                response = requests.post(f"{API_URL}/common-sections", json=payload)

                import pandas as pd
                if response.status_code == 200:
                    data = response.json()

                    st.markdown("### High-Level Differences")
                    st.write(data["differences"])

                    st.markdown("### Primary Commonalities")
                    st.write(data["commonalities"])

                    st.markdown("### Topic Comparison Matrix")

                    matrix_data = []
                    for cluster in data["clusters"]:
                        sources = list(set([c["source_file"] for c in cluster["chunks"]]))
                        matrix_data.append({
                            "Topic Name": cluster["topic"],
                            "Summary": cluster["summary"],
                            "Source Documents": ", ".join(sources)
                        })

                    if matrix_data:
                        df = pd.DataFrame(matrix_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No common topics clustered.")
                else:
                    st.error(f"Backend Error: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

# ── Session History Tab ───────────────────────────────────────────────────────
with tab_history:
    st.header("🕘 Session History")
    st.caption("All previously saved sessions. Click **Restore** to switch to any session without re-uploading.")

    # Refresh button
    if st.button("⟳ Refresh", key="refresh_sessions"):
        st.rerun()

    try:
        hist_res = requests.get(f"{API_URL}/documents/sessions", timeout=5)
        all_sessions = hist_res.json().get("sessions", []) if hist_res.status_code == 200 else []
    except Exception:
        all_sessions = []
        st.error("Could not reach the backend. Make sure the API server is running.")

    if not all_sessions:
        st.info("No saved sessions found. Upload and index some documents to get started.")
    else:
        st.markdown(f"**{len(all_sessions)} saved session(s)**")

        now = datetime.datetime.now()
        avatar_colors = ["#6c6cff", "#ff6c9d", "#6cffb3", "#ffb36c", "#6cc5ff", "#d46cff"]

        # Render 3-column card grid
        cols = st.columns(3)

        for idx, s in enumerate(all_sessions):
            sid = s["session_id"]
            is_active = sid == st.session_state.session_id
            docs = s["documents"]
            chunk_count = s["chunk_count"]
            ts = datetime.datetime.fromtimestamp(s["timestamp"]) if s["timestamp"] else None
            color = avatar_colors[idx % len(avatar_colors)]

            # Relative time
            if ts:
                delta = now - ts
                if delta.total_seconds() < 60:
                    rel_time = "just now"
                elif delta.total_seconds() < 3600:
                    rel_time = f"{int(delta.total_seconds()//60)} min ago"
                elif delta.total_seconds() < 86400:
                    rel_time = f"{int(delta.total_seconds()//3600)} hr ago"
                else:
                    rel_time = ts.strftime("%b %d, %Y  %H:%M")
            else:
                rel_time = "Unknown"

            abs_time = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "—"

            # File pills — all files shown (no truncation in this view)
            pill_html = ""
            for doc in docs:
                pill_html += (
                    f"<span style='display:inline-block;background:#252545;color:#b0b0e0;"
                    f"border-radius:5px;padding:2px 8px;font-size:11px;margin:2px 3px 2px 0;"
                    f"border:1px solid #3a3a60;'>{doc}</span>"
                )

            active_banner = (
                "<div style='background:#6c6cff22;color:#9090ff;border-radius:4px;"
                "padding:2px 8px;font-size:11px;font-weight:600;display:inline-block;"
                "margin-bottom:6px;'>● ACTIVE SESSION</div>"
                if is_active else ""
            )

            border_style = f"2px solid {color}" if is_active else "1px solid #2a2a3c"
            bg_color = "#1a1a2e" if is_active else "#13131d"

            with cols[idx % 3]:
                st.markdown(
                    f"""
                    <div style="border:{border_style};background:{bg_color};border-radius:14px;
                                padding:16px;margin-bottom:12px;min-height:180px;">
                      {active_banner}
                      <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
                        <div style="width:40px;height:40px;border-radius:50%;background:{color};
                                    display:flex;align-items:center;justify-content:center;
                                    font-size:16px;font-weight:800;color:#fff;flex-shrink:0;">
                          {idx+1}
                        </div>
                        <div>
                          <div style="font-size:11px;font-family:monospace;color:#7070a0;
                                      word-break:break-all;">{sid}</div>
                          <div style="font-size:11px;color:#555;margin-top:2px;">🕐 {rel_time} &nbsp;·&nbsp; {abs_time}</div>
                        </div>
                      </div>
                      <div style="margin-bottom:8px;">
                        <span style="font-size:11px;color:#666;font-weight:600;">
                          📄 {len(docs)} file(s) &nbsp;·&nbsp; {chunk_count} chunks
                        </span>
                      </div>
                      <div style="line-height:1.8;">{pill_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                btn_label = "✓ Currently Active" if is_active else f"Restore Session {idx+1}"
                if st.button(btn_label, key=f"hist_restore_{sid}",
                             disabled=is_active, use_container_width=True):
                    st.session_state.session_id = sid
                    st.success(f"✅ Switched to session {idx+1}  —  {len(docs)} doc(s) ready.")
                    st.rerun()
